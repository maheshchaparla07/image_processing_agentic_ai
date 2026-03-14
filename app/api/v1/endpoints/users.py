from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/")
def get_users():
    return {"message": "Get all users endpoint working"}


@router.post("/")
def create_user():
    return {"message": "Create user endpoint working"}