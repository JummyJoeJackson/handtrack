from sqlalchemy.orm import sessionmaker
# Import specific classes directly
from UserSetUp import engine, Lesson, init_db

Session = sessionmaker(bind=engine)
session = Session()

def create_lessons():
    # Initialize tables (Create Lesson table if it doesn't exist)
    init_db()

    # Check for duplicates
    if session.query(Lesson).first():
        print("Lessons already exist. Skipping.")
        return

    # Define the curriculum
    curriculum = [
        {"title": "The Basics", "letters": "A,B,C,D,E", "difficulty": 1},
        {"title": "First Steps", "letters": "F,G,H,I,K", "difficulty": 1},
        {"title": "Mid-Range", "letters": "L,M,N,O,P", "difficulty": 2},
        {"title": "The Twist", "letters": "Q,R,S,T,U", "difficulty": 2},
        {"title": "Advanced", "letters": "V,W,X,Y", "difficulty": 3}
    ]

    print("Populating database with lessons...")
    for data in curriculum:
        lesson = Lesson(
            title=data["title"], 
            target_letters=data["letters"], 
            difficulty_level=data["difficulty"]
        )
        session.add(lesson)
    
    session.commit()
    print("Successfully added 5 lessons!")

if __name__ == "__main__":
    create_lessons()
