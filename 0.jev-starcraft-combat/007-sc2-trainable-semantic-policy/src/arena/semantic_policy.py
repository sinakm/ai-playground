"""Tiny trainable policy for episode 007.

Jev is frozen perception. This module owns the trainable weights.
Architecture: 10 semantic inputs -> 16 tanh hidden units -> 9 action logits.
REINFORCE updates happen once per completed battle.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ACTIONS = ("kite","split","attack","stim","retreat","focus_bane","cover_ally","bait","stutter")
GLOBAL_KEYS = ("baneling_pressure","clumping_danger","encirclement_risk","focus_fire_opportunity","retreat_pressure","formation_instability")
LOCAL_KEYS = ("personal_danger","isolation","escape_pressure","firing_opportunity")
INPUT_DIM = len(GLOBAL_KEYS)+len(LOCAL_KEYS)
HIDDEN_DIM = 16

class SemanticMLP:
    def __init__(self, seed=0, learning_rate=0.003, entropy=0.01):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,0.15,(HIDDEN_DIM,INPUT_DIM)); self.b1=np.zeros(HIDDEN_DIM)
        self.w2=rng.normal(0,0.15,(len(ACTIONS),HIDDEN_DIM)); self.b2=np.zeros(len(ACTIONS))
        self.lr=learning_rate; self.entropy=entropy; self.trajectory=[]

    def vector(self, global_a, local_a):
        return np.asarray([global_a.get(k,0.5) for k in GLOBAL_KEYS]+[local_a.get(k,0.5) for k in LOCAL_KEYS],dtype=float)

    def probs(self,x):
        h=np.tanh(self.w1@x+self.b1); z=self.w2@h+self.b2; z-=z.max()
        p=np.exp(z); p/=p.sum()
        return p,h

    def act(self, global_a, local_a, rng, training=True):
        x=self.vector(global_a,local_a); p,h=self.probs(x)
        idx=int(rng.choice(len(ACTIONS),p=p)) if training else int(np.argmax(p))
        if training: self.trajectory.append((x,h,p,idx))
        return ACTIONS[idx], float(p[idx]), {a:float(p[i]) for i,a in enumerate(ACTIONS)}

    def finish_episode(self,reward):
        if not self.trajectory: return
        # Episodic REINFORCE. Same terminal reward is assigned to all decisions.
        # Normalize reward to roughly [-1,1] around a mediocre fight.
        advantage=float(np.clip((reward-15.0)/5.0,-1.0,1.0))
        gw1=np.zeros_like(self.w1); gb1=np.zeros_like(self.b1); gw2=np.zeros_like(self.w2); gb2=np.zeros_like(self.b2)
        n=len(self.trajectory)
        for x,h,p,idx in self.trajectory:
            d=-p.copy(); d[idx]+=1.0
            # Small entropy-like push prevents premature collapse.
            d += self.entropy*(1.0/len(ACTIONS)-p)
            d*=advantage
            gw2+=np.outer(d,h); gb2+=d
            dh=(self.w2.T@d)*(1-h*h)
            gw1+=np.outer(dh,x); gb1+=dh
        scale=self.lr/max(1,n)
        self.w1+=scale*gw1; self.b1+=scale*gb1; self.w2+=scale*gw2; self.b2+=scale*gb2
        self.trajectory.clear()

    def save(self,path):
        Path(path).write_text(json.dumps({"w1":self.w1.tolist(),"b1":self.b1.tolist(),"w2":self.w2.tolist(),"b2":self.b2.tolist()}),encoding="utf-8")

    def load(self,path):
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        self.w1=np.asarray(d["w1"]); self.b1=np.asarray(d["b1"]); self.w2=np.asarray(d["w2"]); self.b2=np.asarray(d["b2"])
