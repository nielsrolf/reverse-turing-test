# Turing Test Reverse

This repo contains an implementation of a turing test inspired game - however, GPT is the judge and needs to recognize itself among other LLMs and optionally a human subject.

## How to play
The test is implemented as a [discord bot](https://discord.gg/QApvYrsw). In order to play, type `/turing` and fill out the form that shows up.

## Limitations
This is a quick implementation that lacks many things that would be cool, eg:
- let humans and LLMs that are not from open ai play as judge (currently the judge is implemented using function calls, so one would need to implement some response parsing as alternative for other LLMs)