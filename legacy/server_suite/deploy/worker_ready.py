#!/usr/bin/env python3
"""Minimal readiness agent used by the deployment harness.

It deliberately does not accept arbitrary remote commands.  The master still
launches the benchmark/runtime through SSH or the configured cluster launcher.
"""
from __future__ import annotations
import argparse,json,socket


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=29600);a=p.parse_args()
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((a.host,a.port));s.listen()
        while True:
            c,_=s.accept()
            with c:
                data=c.recv(4096).decode(errors='replace').strip()
                reply={'ready':True,'service':'routersense-worker-ready','request':data[:128]}
                c.sendall((json.dumps(reply)+'\n').encode())
if __name__=='__main__':main()
