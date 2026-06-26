"""Headless web-serving layer: drive the simulation in a background thread and
expose live state over a small HTTP API (see ``app``) for the LAN dashboard.

This package is dependency-light at import time on purpose — ``app``/``__main__``
pull in FastAPI/uvicorn (the optional ``serve`` extra), while ``runner`` only
needs the core simulation and torch.
"""
