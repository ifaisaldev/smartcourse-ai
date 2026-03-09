import pandas as pd
import random
import os

def generate_synthetic_data(num_courses=100):
    departments = ['Computer Science', 'Data Science', 'Business', 'Arts', 'Engineering', 'Mathematics', 'Physics', 'Marketing']
    difficulties = ['Beginner', 'Intermediate', 'Advanced']
    universities = ['MIT', 'Stanford', 'Harvard', 'Berkeley', 'Oxford', 'Cambridge', 'Caltech', 'Columbia']
    
    topics = [
        "Python Programming", "Machine Learning", "Deep Learning", "Data Visualization", 
        "Web Development", "React JS", "Flask API", "Business Strategy", "Digital Marketing",
        "Linear Algebra", "Quantum Mechanics", "History of Art", "Creative Writing",
        "Financial Accounting", "Project Management", "Cybersecurity", "Cloud Computing",
        "Blockchain Basics", "Artificial Intelligence", "Natural Language Processing"
    ]
    
    courses = []
    
    for i in range(num_courses):
        topic = random.choice(topics)
        dept = random.choice(departments)
        
        # Make description somewhat relevant to topic
        desc_templates = [
            f"Learn the fundamentals of {topic} in this comprehensive course.",
            f"Master {topic} with real-world projects and examples.",
            f"An advanced guide to {topic} for professionals.",
            f"Introduction to {topic} covering key concepts and techniques.",
            f"Explore the depths of {topic} and its applications in {dept}."
        ]
        
        course = {
            'course_title': f"{topic} - {random.choice(['Fundamentals', 'Masterclass', 'Bootcamp', 'Specialization', '101'])}",
            'university': random.choice(universities),
            'difficulty_level': random.choice(difficulties),
            'rating': round(random.uniform(3.5, 5.0), 1),
            'description': random.choice(desc_templates),
            'department': dept
        }
        courses.append(course)
        
    df = pd.DataFrame(courses)
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    
    output_path = 'data/raw/courses.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {num_courses} synthetic courses at {output_path}")

if __name__ == "__main__":
    # Generating a smaller set for quick testing, but structure supports 8500+
    generate_synthetic_data(num_courses=200) 
