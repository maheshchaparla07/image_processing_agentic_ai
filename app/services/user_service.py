from sqlalchemy.orm import Session

from app.repositories.user_repo import UserRepository


class UserService:
	def __init__(self, db: Session):
		self.repo = UserRepository(db)

	def list_users(self):
		return self.repo.get_all()


