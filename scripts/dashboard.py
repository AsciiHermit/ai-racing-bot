"""Launch the setup dashboard: Track -> Vehicle -> Physics -> Agent ->
Simulation. Requires the `viz` extra (pip install -e ".[viz]").

    python scripts/dashboard.py
"""
from ars.dashboard import DashboardApp

if __name__ == "__main__":
    DashboardApp().run()
