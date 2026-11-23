#!/usr/bin/env python3
"""Small API warmup outside measured window. Avoids conflating cold request with pressure."""
import argparse
import asyncio
import aiohttp

async def main(args):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async def single(i):
            prompt=("A short reproducible warmup sentence. " * (4 if i%2==0 else 16))
            body={"model":args.model,"messages":[{"role":"user","content":prompt}],
                  "temperature":0,"max_completion_tokens":32,"ignore_eos":True}
            async with session.post(args.url.rstrip('/')+'/v1/chat/completions',json=body) as resp:
                if resp.status!=200:
                    raise RuntimeError(f'warmup HTTP {resp.status}: {(await resp.text())[:200]}')
                result=await resp.json()
                if not result.get('choices'):raise RuntimeError('empty warmup result')
        for batch in range(2):
            await asyncio.gather(*(single(i+8*batch) for i in range(8)))
    print('[PASS] 16 warmup requests; not included in benchmark window')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--model',default='Qwen/Qwen3-0.6B')
    args=ap.parse_args();asyncio.run(main(args))
