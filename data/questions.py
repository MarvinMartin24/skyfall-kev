"""The typed questions asked of every state. Shared by the generator and the client."""
QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "You are the skydiver Y falling through the grid. Obstacles scroll up toward you. Which way should you move this tick to avoid hitting anything?",
        "criteria": {
            "left": "move one column to the left",
            "right": "move one column to the right",
            "stay": "do not move sideways",
        },
    },
    "open": {
        "type": "noul",
        "instructions": "Should the skydiver open the parachute this tick? Opening is only allowed at or below 1000 m altitude and only when the cell directly below is empty.",
    },
}
