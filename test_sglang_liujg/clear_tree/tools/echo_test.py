import asyncio, socket, subprocess, sys, time
async def echo(r, w):
    while data := await r.read(1 << 16):
        w.write(data); await w.drain()
async def main(rtt):
    srv = await asyncio.start_server(echo, "127.0.0.1", 29101)
    proc = subprocess.Popen([sys.executable, "delay_proxy.py", "--listen-port", "29100",
        "--upstream-host", "127.0.0.1", "--upstream-port", "29101", "--rtt-ms", str(rtt)],
        stdout=subprocess.DEVNULL)
    await asyncio.sleep(0.8)
    r, w = await asyncio.open_connection("127.0.0.1", 29100)
    ts = []
    for _ in range(50):
        t = time.perf_counter(); w.write(b"x" * 2000); await w.drain()
        got = 0
        while got < 2000: got += len(await r.read(1 << 16))
        ts.append((time.perf_counter() - t) * 1e3)
    proc.terminate(); srv.close()
    ts.sort(); print(f"rtt={rtt} median={ts[25]:.2f} p90={ts[45]:.2f} max={ts[-1]:.2f} ms")
for rtt in (0, 10, 40):
    asyncio.run(main(rtt))
