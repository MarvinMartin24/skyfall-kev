"""The typed questions asked of every state. Shared by the generator and the client."""
QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "You are the skydiver Y falling through the grid; everything scrolls up toward you one row per tick. Clouds (C) are harmless and you fly straight through them. Any other letter is a deadly obstacle. Which way should you move this tick to avoid hitting anything deadly?",
        "criteria": {
            "left": "move one column to the left",
            "right": "move one column to the right",
            "stay": "do not move sideways",
        },
    },
    "open": {
        "type": "noul",
        "instructions": "Should the skydiver open the parachute this tick? Opening is only allowed at or below 1000 m altitude and only when the cell directly below is empty or a cloud (C).",
    },
}
