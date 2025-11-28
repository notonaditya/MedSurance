from app import app, db, User

def update_username():
    with app.app_context():
        # Find the user
        user = User.query.filter_by(username='Abhishek').first()
        if user:
            # Update the username
            user.username = 'abhishek'
            db.session.commit()
            print("Username updated successfully")
        else:
            print("User not found")

if __name__ == "__main__":
    update_username()