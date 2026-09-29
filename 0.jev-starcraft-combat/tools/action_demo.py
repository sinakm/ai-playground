"""Force one action on every Marine for 2 s and screenshot the SC2 window.

Used for the field guide's action gallery. Run from an episode folder (004+):
    uv run python ../tools/action_demo.py <out_dir> spread clump kite bait
Realtime mode on purpose: SC2's debug chat text fades on wall-clock time.
No Jev calls, so no cost.
"""
import subprocess, sys, tempfile
from pathlib import Path
from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from arena import config
from arena.bot import ArenaBot
from arena.policies import Decision

OUT = Path(sys.argv[1]); SQUAD = {"spread", "clump", "retreat", "attack", "stim"}

def shot(name):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "gdigrab", "-i", "title=StarCraft II",
                    "-frames:v", "1", str(OUT / f"{name}.png")], check=False)

class Fixed:
    uses_blackboard = False
    def __init__(self, action): self.action = action; self.name = "demo_" + action
    def decide(self, state):
        if self.action in SQUAD:
            return Decision(action=self.action)
        tags = [m["id"] for m in state["marines"]]
        return Decision(action=self.action, marine_actions={t: self.action for t in tags})

class Demo(ArenaBot):
    def __init__(self, action, first, **kw):
        super().__init__(**kw); self.action = action; self.first = first; self.done = set()
    async def on_step(self, iteration):
        await super().on_step(iteration)
        loop = self.state.game_loop
        sp = getattr(self, "spawn_loop", None)
        if self.first and sp is not None and "units" not in self.done and loop >= sp + 420:
            self.done.add("units"); shot("units_prefight")
        if self.start_loop is None: return
        e = loop - self.start_loop
        if self.first and "t0" not in self.done and e >= 0:
            self.done.add("t0"); shot("before")
        if "t1" not in self.done and e >= 45:
            self.done.add("t1"); shot(self.action)
            self.finished = True
            await self.client.leave()

for i, a in enumerate(sys.argv[2:]):
    d = Path(tempfile.mkdtemp())
    bot = Demo(a, i == 0 and a == "spread", policy=Fixed(a), run_dir=d, seed=0, realtime=True)
    try:
        run_game(maps.get(config.MAP_NAME), [Bot(Race.Terran, bot), Computer(Race.Zerg, Difficulty.VeryEasy)],
                 realtime=True, random_seed=0)
    except Exception as ex:
        print("run error", a, type(ex).__name__)
    finally:
        bot.close()
    print("done", a)
