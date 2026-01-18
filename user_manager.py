from sqlalchemy.orm import sessionmaker
from UserSetUp import engine, User, UserProgress, Lesson, init_db
from datetime import datetime
import time

# Setup connection
Session = sessionmaker(bind=engine)
session = Session()

# --- HELPER FUNCTIONS ---

def format_username(name):
    return name.strip().title()

def get_user(username):
    clean_name = format_username(username)
    return session.query(User).filter_by(username=clean_name).first()

def create_user(username):
    clean_name = format_username(username)
    if get_user(clean_name):
        print(f"⚠️  User '{clean_name}' already exists.")
        return get_user(clean_name)

    new_user = User(username=clean_name)
    session.add(new_user)
    session.commit()
    print(f"✅ Created new user: {clean_name}")
    return new_user

def delete_user(username):
    user = get_user(username)
    if not user:
        print(f"❌ User '{username}' not found.")
        return False
    
    print(f"\n⚠️  WARNING: Are you sure you want to delete user '{user.username}' and all associated data?")
    choice = input("   Type 'yes' to confirm: ").lower()
    
    if choice != 'yes':
        print("   Deletion cancelled.")
        return False
    
    session.query(UserProgress).filter_by(user_id=user.id).delete()
    session.delete(user)
    session.commit()
    print(f"🗑️  Deleted user: {user.username}.")
    return True

def print_user_stats(user):
    session.refresh(user)
    print("\n" + "="*30)
    print(f"👤  PLAYER: {user.username}")
    print("-" * 30)
    print(f"🔥  Current Streak:   {user.current_streak} days")
    print(f"⭐  Total XP:         {user.total_xp}")
    print(f"📅  Last Active:      {user.last_practice_date.strftime('%Y-%m-%d %H:%M')}")
    print("="*30 + "\n")

# --- LESSON MANAGEMENT ---

def get_all_lessons():
    """Returns a list of all available lessons."""
    return session.query(Lesson).order_by(Lesson.id).all()

def select_lesson_menu():
    """Interactive menu to pick a lesson from the database."""
    lessons = get_all_lessons()
    if not lessons:
        print("   No lessons found in database.")
        return None

    print("\n   --- AVAILABLE LESSONS ---")
    for l in lessons:
        print(f"   {l.id}. {l.title} (Targets: {l.target_letters})")
    
    try:
        choice = int(input("\n   Enter Lesson ID to select: "))
        selected = session.query(Lesson).filter_by(id=choice).first()
        if selected:
            return selected
        else:
            print("   Invalid ID.")
            return None
    except ValueError:
        print("   Invalid input.")
        return None

def check_lesson_stats(user):
    """Detailed mastery view for a specific lesson."""
    lesson = select_lesson_menu()
    if not lesson: return

    stats = get_lesson_status(user, lesson)
    print(f"\n   --- MASTERY: {lesson.title} ---")
    for letter, score in stats.items():
        # Visual bar: "A: 50% [█████     ]"
        bar_len = score // 10
        bar = "█" * bar_len + "." * (10 - bar_len)
        print(f"   {letter}: {score:3}% [{bar}]")
    
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

# --- INTELLIGENT TEACHER LOGIC ---

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

# --- GAMEPLAY ---

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
        print(f"   ✅ Correct! +{xp_gain} XP | '{letter}' Mastery: {progress.mastery_score}%")
    else:
        progress.mastery_score = max(0, progress.mastery_score - 10)
        print(f"   ❌ Incorrect. '{letter}' Mastery: {progress.mastery_score}%")

    session.commit()

# --- MENUS ---

def login():
    print("\n👋 Welcome to the ASL Trainer!")
    raw_name = input("   Please enter your name: ")
    if not raw_name: return None

    user = get_user(raw_name)
    if user:
        print(f"\n   Welcome back, {user.username}!")
    else:
        print(f"\n   User '{format_username(raw_name)}' not found.")
        choice = input("   Create new account? (y/n): ").lower()
        if choice == 'y': user = create_user(raw_name)
        else: return None
    return user