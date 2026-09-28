# ai-playground

One small, reproducible project per week with a new AI model that is fun to
watch. Every folder is one episode. Each episode runs on its own: clone the
repo, open the folder, follow its README.

No hype. Each README reports measured results, cost, latency, and what failed.

## Episodes

| # | Episode | Model | Video |
|---|---------|-------|-------|
| 001 | [SC2 combat arena](001-sc2-combat-arena/) | TypeSafe Jev | [video](https://github.com/sinakm/ai-playground/releases/tag/ep001) |
| 002 | [SC2 squad with goal](002-sc2-squad-with-goal/) | TypeSafe Jev | [video](https://github.com/sinakm/ai-playground/releases/tag/ep002) |
| 003 | [SC2 marine micro](003-sc2-marine-micro/) | TypeSafe Jev | [video](https://github.com/sinakm/ai-playground/releases/tag/ep003) |
| 004 | [SC2 squad commander](004-sc2-squad-commander/) | TypeSafe Jev | [video](https://github.com/sinakm/ai-playground/releases/tag/ep004) |

Start with 001; each episode builds on the previous one.

## Running an episode

1. Install [uv](https://docs.astral.sh/uv/).
2. `cd NNN-<slug>`
3. `cp .env.example .env` and fill in the keys the episode lists.
4. Follow the episode README.

## License

Code is MIT. Third-party models, games, and assets keep their own licenses;
each episode lists them under "Credits and licenses".
