#!/usr/bin/env python3
"""User-space TCP delay proxy. No root needed.

Target --opt-tree-addr/--opt-tree-port point at this proxy; the proxy
connects to the Draft's ROUTER port. Each direction gets rtt/2 one-way
delay, optional uniform jitter and an optional bandwidth cap. Bytes keep
their order. Timer resolution is about 1 ms, so log the measured R.

  python3 delay_proxy.py --listen-port 29000 \
      --upstream-host <draft-ip> --upstream-port <draft-port> --rtt-ms 20
"""
import argparse
import asyncio
import random
import socket
import time


def _nodelay(writer):
    sock = writer.get_extra_info("socket")
    if sock is not None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


async def _pipe(reader, writer, one_way_s, jitter_s, bits_per_s, name, stats):
    queue = asyncio.Queue()

    async def rx():
        while True:
            data = await reader.read(1 << 16)
            queue.put_nowait((time.monotonic(), data))
            if not data:
                return

    async def tx():
        link_free = 0.0
        while True:
            arrived, data = await queue.get()
            if not data:
                writer.close()
                return
            due = arrived + one_way_s
            if jitter_s:
                due += random.uniform(0.0, jitter_s)
            if bits_per_s:
                start = max(due, link_free)
                link_free = start + len(data) * 8.0 / bits_per_s
                due = link_free
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(data)
            await writer.drain()
            stats[name] += len(data)

    await asyncio.gather(rx(), tx())


async def _handle(client_r, client_w, args):
    try:
        up_r, up_w = await asyncio.open_connection(
            args.upstream_host, args.upstream_port
        )
    except OSError as exc:
        # Upstream not listening yet; the ZMQ DEALER reconnects on its own.
        print(f"[proxy] upstream connect failed: {exc}", flush=True)
        client_w.close()
        return
    _nodelay(client_w)
    _nodelay(up_w)
    one_way = args.rtt_ms / 2000.0
    jitter = args.jitter_ms / 2000.0
    bits = args.bw_mbps * 1e6 if args.bw_mbps else 0.0
    stats = {"t2d": 0, "d2t": 0}
    print(f"[proxy] connection, one-way {one_way * 1e3:.2f} ms", flush=True)
    try:
        await asyncio.gather(
            _pipe(client_r, up_w, one_way, jitter, bits, "t2d", stats),
            _pipe(up_r, client_w, one_way, jitter, bits, "d2t", stats),
        )
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        print(f"[proxy] closed, bytes {stats}", flush=True)
        for w in (client_w, up_w):
            w.close()


async def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--upstream-host", required=True)
    p.add_argument("--upstream-port", type=int, required=True)
    p.add_argument("--rtt-ms", type=float, default=0.0, help="added RTT")
    p.add_argument("--jitter-ms", type=float, default=0.0, help="added RTT jitter")
    p.add_argument("--bw-mbps", type=float, default=0.0, help="0 = no cap")
    args = p.parse_args()
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, args), args.listen_host, args.listen_port
    )
    print(f"[proxy] {args.listen_host}:{args.listen_port} -> "
          f"{args.upstream_host}:{args.upstream_port} +{args.rtt_ms} ms RTT", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
