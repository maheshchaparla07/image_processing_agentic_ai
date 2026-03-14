import secrets


def generate_api_token() -> str:
	return secrets.token_urlsafe(32)


