import json, time, random
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body["sampling_params"]["max_new_tokens"]
        ids = [(sum(body["input_ids"]) + i) % 1000 for i in range(n)]
        top = [[[-0.1, t, None], [-0.1 - (i % 7) * 0.4, t + 1, None]] for i, t in enumerate(ids)]
        meta = {"id": "r%d" % random.randint(0, 10**9), "completion_tokens": n,
                "output_token_logprobs": [[-0.1, t, None] for t in ids], "output_top_logprobs": top}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for c in range(1, n + 1, 4):
                time.sleep(0.002)
                self.wfile.write(("data: " + json.dumps({"meta_info": {"id": meta["id"], "completion_tokens": c}}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = json.dumps({"text": "", "output_ids": ids, "meta_info": meta}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
ThreadingHTTPServer(("127.0.0.1", 29201), H).serve_forever()
