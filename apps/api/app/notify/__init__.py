"""Patient alerts: WhatsApp messages and voice calls through Exotel.

This package is deliberately separate from sign-in. It never sees a user session:
it reads what is due from the database with a server key, sends, and writes the
result back.
"""
