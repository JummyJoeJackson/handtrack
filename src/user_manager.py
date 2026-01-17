from sqlalchemy.orm import sessionmaker
from UserSetUp import engine, User, UserProgress, init_db
from datetime import datetime, timedelta

Session = sessionmaker(bind=engine)
session = Session()


def get_user(username):
    """Finds a user by name (case-insensitive due to formatting)."""

    clean_name = username.strip().upper()
    return session.query(User).filter_by(username=clean_name).first()

def create_user(username):
    """Creates a new user."""
    clean_name = username.strip().upper()
    
    if get_user(clean_name):
        print(f"User '{clean_name}' already exists.")
        return get_user(clean_name)

    new_user = User(username=clean_name)
    session.add(new_user)
    session.commit()
    print(f"Created new user: {clean_name}")
    return new_user

def delete_user(username):
    """Deletes a user and their progress."""

    user = get_user(username)
    
    if not user:
        print(f"User '{username}' not found.")
        return False
    
    session.query(UserProgress).filter_by(user_id=user.id).delete()
    session.delete(user)
    session.commit()
    print(f"Deleted user: {user.username} and all their data.")
    return True

def print_user_stats(user):
    """Displays the user's dashboard."""

    print("\n" + "="*30)
    print(f" PLAYER: {user.username}")
    print("=" * 30)
    print(f"Current Streak:   {user.current_streak} days")
    print(f"Total XP:         {user.total_xp}")
    print(f"Last Active:      {user.last_practice_date.strftime('%Y-%m-%d %H:%M')}")
    print("="*30 + "\n")

def login():
    """The login screen."""

    print("\n Welcome to the ASL Trainer!")
    raw_name = input("Please enter your name to sign in: ")

    while not raw_name:
        raw_name = input("Name cannot be empty. Please enter your name: ")

    user = get_user(raw_name)
    
    if user:
        print(f"\n Welcome back, {user.username}!")
    else:
        print(f"\n User '{raw_name.strip().upper()}' not found.")
        choice = input("   Create new account? (y/n): ").lower()
        if choice == 'y':
            user = create_user(raw_name)
        else:
            print("Login cancelled.")
            return None

    print_user_stats(user)
    return user

def update_streak(user):
    """ Updates the user's practice streak based on last practice date. """

    now = datetime.now()
    last = user.last_practice_date
    
    diff_days = (now.date() - last.date()).days

    if diff_days == 0:
        pass
    elif diff_days == 1:
        user.current_streak += 1
    else:
        user.current_streak = 1
    
    user.last_practice_date = now

def record_attempt(username, letter, is_correct):
    """ Records a user's attempt at a letter and updates XP, streak, and mastery. """

    user = get_user(username)
    if not user:
        print("User not found!")
        return

    # Find existing progress, if there is none, create new
    progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
    
    if not progress:
        progress = UserProgress(user_id=user.id, letter=letter, mastery_score=0)
        session.add(progress)
    
    if is_correct:
        xp_gain = 10
        user.total_xp += xp_gain
        
        update_streak(user)
        
        progress.mastery_score = min(100, progress.mastery_score + 10)
        
        print(f"Correct! +{xp_gain} XP | Streak: {user.current_streak} | '{letter}' Mastery: {progress.mastery_score}%")
    else:
        progress.mastery_score = max(0, progress.mastery_score - 5)
        print(f"Incorrect. '{letter}' Mastery: {progress.mastery_score}%")

    session.commit()

# --- TEST ---
if __name__ == "__main__":
    init_db()

    current_user = login()

    if current_user:
        cmd = input("Type 'delete' to delete this account, or anything else to exit: ")
        if cmd.lower() == 'delete':
            delete_user(current_user.username)


    """
    me = create_user("Student")
    
    print("\n--- Attempt 1: Correct 'A' ---")
    record_attempt("Student", "A", True)

    print("\n--- Attempt 2: Incorrect 'B' ---")
    record_attempt("Student", "B", False)

    refreshed_user = get_user("Student")
    print(f"\nFINAL STATS: XP={refreshed_user.total_xp}, Streak={refreshed_user.current_streak}")
    """