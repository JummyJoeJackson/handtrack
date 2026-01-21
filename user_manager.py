from datetime import datetime
from sqlalchemy.orm import sessionmaker
from UserSetUp import engine, User, UserProgress, Lesson, init_db
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

console = Console()


Session = sessionmaker(bind=engine)
session = Session()


def format_username(name):
    return name.strip().title()


def get_user(username):
    clean_name = format_username(username)
    return session.query(User).filter_by(username=clean_name).first()


def create_user(username):
    clean_name = format_username(username)
    if get_user(clean_name):
        console.print(f"⚠️  [warning]User '{clean_name}' already exists.[/warning]")
        return get_user(clean_name)

    new_user = User(username=clean_name)
    session.add(new_user)
    session.commit()
    console.print(f"✅ [success]Created new user: {clean_name}[/success]")
    return new_user


def delete_user(username):
    user = get_user(username)
    if not user:
        console.print(f"❌ [error]User '{username}' not found.[/error]")
        return False
    
    console.print(f"\n⚠️  [bold red]WARNING: Are you sure you want to delete user '{user.username}' and all associated data?[/bold red]")
    if not Confirm.ask("   Confirm deletion?"):
        console.print("   [info]Deletion cancelled.[/info]")
        return False
    
    session.query(UserProgress).filter_by(user_id=user.id).delete()
    session.delete(user)
    session.commit()
    console.print(f"🗑️  [success]Deleted user: {user.username}.[/success]")
    return True


def print_user_stats(user):
    session.refresh(user)
    table = Table(title=f"👤 PLAYER: {user.username}", show_header=True, header_style="bold magenta")
    table.add_column("Stat", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("🔥 Current Streak", f"{user.current_streak} days")
    table.add_row("⭐ Total XP", str(user.total_xp))
    table.add_row("📅 Last Active", user.last_practice_date.strftime('%Y-%m-%d %H:%M'))
    
    console.print(table)


def get_all_lessons():
    """Returns a list of all available lessons."""
    return session.query(Lesson).order_by(Lesson.id).all()


def select_lesson_menu():
    """Interactive menu to pick a lesson from the database."""
    lessons = get_all_lessons()
    if not lessons:
        console.print("   [warning]No lessons found in database.[/warning]")
        return None

    table = Table(title="AVAILABLE LESSONS", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Targets", style="yellow")
    
    for l in lessons:
        table.add_row(str(l.id), l.title, l.target_letters)
    
    console.print(table)
    
    choice = Prompt.ask("Enter Lesson ID to select", choices=[str(l.id) for l in lessons] + ["q"])
    if choice == "q": return None
    
    selected = session.query(Lesson).filter_by(id=int(choice)).first()
    return selected


def check_lesson_stats(user):
    """Detailed mastery view for a specific lesson."""
    lesson = select_lesson_menu()
    if not lesson: return

    stats = get_lesson_status(user, lesson)
    table = Table(title=f"MASTERY: {lesson.title}", show_header=True, header_style="bold magenta")
    table.add_column("Letter", style="cyan")
    table.add_column("Mastery", style="green")
    table.add_column("Progress Bar", style="yellow")
    
    for letter, score in stats.items():
        bar_len = score // 10
        bar = "█" * bar_len + "░" * (10 - bar_len)
        table.add_row(letter, f"{score}%", f"[{bar}]")
    
    console.print(table)
    input("\n   Press Enter to continue...")


def skip_current_lesson(user):
    lesson = get_next_lesson(user)
    if not lesson:
        print("   You have already finished everything!")
        return

    print(f"   Skipping '{lesson.title}'...")
    letters = lesson.target_letters.split(',')
    
    for letter in letters:
        progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
        if not progress:
            progress = UserProgress(user_id=user.id, letter=letter)
            session.add(progress)
        progress.mastery_score = 100
    
    session.commit()
    print(f"   ✅ Lesson '{lesson.title}' marked as complete!")


def get_next_lesson(user):
    all_lessons = session.query(Lesson).order_by(Lesson.id).all()
    for lesson in all_lessons:
        letters = lesson.target_letters.split(',')
        completed_count = 0
        for letter in letters:
            progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
            if progress and progress.mastery_score >= 80:
                completed_count += 1
        if completed_count < len(letters):
            return lesson
    return None 


def get_lesson_status(user, lesson):
    status = {}
    letters = lesson.target_letters.split(',')
    for letter in letters:
        progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
        status[letter] = progress.mastery_score if progress else 0
    return status


def record_attempt(username, letter, is_correct):
    user = get_user(username)
    if not user: return

    progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
    if not progress:
        progress = UserProgress(user_id=user.id, letter=letter, mastery_score=0)
        session.add(progress)
    
    if is_correct:
        xp_gain = 10
        user.total_xp += xp_gain
        now = datetime.now()
        if (now.date() - user.last_practice_date.date()).days == 1:
            user.current_streak += 1
        elif (now.date() - user.last_practice_date.date()).days > 1:
            user.current_streak = 1
        user.last_practice_date = now

        progress.mastery_score = min(100, progress.mastery_score + 20) 
        console.print(f"   ✅ [success]Correct! +{xp_gain} XP | '{letter}' Mastery: {progress.mastery_score}%[/success]")
    else:
        progress.mastery_score = max(0, progress.mastery_score - 10)
        console.print(f"   ❌ [error]Incorrect. '{letter}' Mastery: {progress.mastery_score}%[/error]")

    session.commit()


def login():
    console.print(Panel("[bold cyan]Welcome to SignCLI![/bold cyan]", expand=False))
    raw_name = Prompt.ask("Please enter your name")
    if not raw_name: return None

    user = get_user(raw_name)
    if user:
        console.print(f"\n   [info]Welcome back, {user.username}![/info]")
    else:
        console.print(f"\n   [warning]User '{format_username(raw_name)}' not found.[/warning]")
        if Confirm.ask("   Create new account?"):
            user = create_user(raw_name)
        else:
            return None
    return user
