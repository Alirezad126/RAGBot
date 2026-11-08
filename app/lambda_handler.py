"""Lambda entry point. Mangum translates the function URL's event into ASGI."""

from mangum import Mangum

from app.main import app

handler = Mangum(app, lifespan="off")
