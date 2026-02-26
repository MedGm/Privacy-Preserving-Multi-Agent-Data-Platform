import asyncio
import spade
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

async def main():
    try:
        agent = spade.agent.Agent("debug@xmpp-server", "password")
        agent.verify_security = False
        print("Starting agent...")
        await agent.start(auto_register=True)
        print("Agent started successfully!")
        await agent.stop()
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(main())
