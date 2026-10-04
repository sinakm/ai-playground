"""Tiny trainable policy for episode 007.

Jev is frozen perception. This module owns the trainable weights.
Architecture: 10 semantic inputs -> 16 tanh hidden units -> 9 action logits.
REINFORCE updates happen once per completed battle, against a running reward baseline,
using only decisions whose sampled action was actually executed.
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
    def __init__(self, seed=0, learning_rate=0.05, entropy=0.01, baseline_alpha=0.1):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,0.15,(HIDDEN_DIM,INPUT_DIM)); self.b1=np.zeros(HIDDEN_DIM)
        self.w2=rng.normal(0,0.15,(len(ACTIONS),HIDDEN_DIM)); self.b2=np.zeros(len(ACTIONS))
        self.lr=learning_rate; self.entropy=entropy; self.trajectory=[]
        # Running reward baseline (EMA of reward and squared deviation), persisted with the weights.
        self.baseline_alpha=baseline_alpha; self.baseline=None; self.reward_var=25.0; self.episodes=0

    def vector(self, global_a, local_a):
        return np.asarray([global_a.get(k,0.5) for k in GLOBAL_KEYS]+[local_a.get(k,0.5) for k in LOCAL_KEYS],dtype=float)

    def probs(self,x):
        h=np.tanh(self.w1@x+self.b1); z=self.w2@h+self.b2; z-=z.max()
        p=np.exp(z); p/=p.sum()
        return p,h

    def act(self, global_a, local_a, rng, training=True, key=None):
        x=self.vector(global_a,local_a); p,h=self.probs(x)
        idx=int(rng.choice(len(ACTIONS),p=p)) if training else int(np.argmax(p))
        if training: self.trajectory.append((x,h,p,idx,key))
        return ACTIONS[idx], float(p[idx]), {a:float(p[i]) for i,a in enumerate(ACTIONS)}

    def drop_unexecuted(self, start, executed):
        """Keep only decisions from trajectory[start:] whose sampled action actually ran.
        `executed` maps key (Marine tag) -> executed action; reflexes and code rules can replace
        the sampled action, and those steps carry no information about the network's choice."""
        if executed is None: return
        tail=[t for t in self.trajectory[start:] if executed.get(t[4])==ACTIONS[t[3]]]
        del self.trajectory[start:]
        self.trajectory.extend(tail)

    def advantage(self, reward):
        """Reward relative to the running baseline, in units of the running std. First episode: 0."""
        if self.baseline is None:
            return 0.0
        return float(np.clip((reward-self.baseline)/max(1.0, math.sqrt(self.reward_var)), -3.0, 3.0))

    def update_baseline(self, reward):
        if self.baseline is None:
            self.baseline=float(reward)
        else:
            dev=reward-self.baseline
            self.baseline+=self.baseline_alpha*dev
            self.reward_var=(1-self.baseline_alpha)*self.reward_var+self.baseline_alpha*dev*dev
        self.episodes+=1

    def finish_episode(self,reward):
        advantage=self.advantage(reward)
        self.update_baseline(reward)
        if not self.trajectory or advantage==0.0:
            self.trajectory.clear(); return advantage
        # Episodic REINFORCE with a running baseline. Same advantage for every kept decision.
        gw1=np.zeros_like(self.w1); gb1=np.zeros_like(self.b1); gw2=np.zeros_like(self.w2); gb2=np.zeros_like(self.b2)
        n=len(self.trajectory)
        for x,h,p,idx,_ in self.trajectory:
            d=-p.copy(); d[idx]+=1.0
            d*=advantage
            # Small entropy-like push toward uniform prevents premature collapse (not scaled by advantage).
            d += self.entropy*(1.0/len(ACTIONS)-p)
            gw2+=np.outer(d,h); gb2+=d
            dh=(self.w2.T@d)*(1-h*h)
            gw1+=np.outer(dh,x); gb1+=dh
        scale=self.lr/max(1,n)
        self.w1+=scale*gw1; self.b1+=scale*gb1; self.w2+=scale*gw2; self.b2+=scale*gb2
        self.trajectory.clear()
        return advantage

    def save(self,path):
        Path(path).write_text(json.dumps({"w1":self.w1.tolist(),"b1":self.b1.tolist(),"w2":self.w2.tolist(),"b2":self.b2.tolist(),
                                          "baseline":self.baseline,"reward_var":self.reward_var,"episodes":self.episodes}),encoding="utf-8")

    def load(self,path):
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        self.w1=np.asarray(d["w1"]); self.b1=np.asarray(d["b1"]); self.w2=np.asarray(d["w2"]); self.b2=np.asarray(d["b2"])
        self.baseline=d.get("baseline"); self.reward_var=d.get("reward_var",25.0); self.episodes=d.get("episodes",0)
