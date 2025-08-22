from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import random
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED = 123
random.seed(SEED)
np.random.seed(SEED)


class VEvent(Enum):
    ELIGIBLE = auto()   # agente pasa a elegible
    SLOT_OPEN = auto()  # clínica anuncia cupo

@dataclass(order=True)
class PQItem:
    # Heap de prioridad (menor key = más prioridad)
    key: tuple
    serial: int
    aid: int = field(compare=False)

@dataclass
class Agent:
    id: int
    age: int
    role: str               # 'healthcare' | 'teacher' | 'other'
    subpop: str             # 'A' | 'B' | 'C'
    risk_score: float       # 0-10
    access_level: int       # 1=alto, 2=medio, 3=bajo (más barrera)
    x: float
    y: float
    reticence: float        # 0..1 (propensión a no presentarse)
    ts_eligible: float = None
    vaccinated: bool = False
    dose_num: int = 0
    reminders_sent: int = 0
    reachable: Optional[set] = None  # clínicas alcanzables
    nearest_km: Optional[float] = None  # distancia a la más cercana

@dataclass
class Clinic:
    id: int
    x: float
    y: float
    mean_daily_slots: int
    reserved_low_access_frac: float = 0.25  # % de cupos reservados a access=3


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5)

def tier(agent: Agent) -> int:
    # 1: ≥65, sanitario o docente; 2: alto riesgo; 3: resto
    if agent.age >= 65 or agent.role in {"healthcare", "teacher"}:
        return 1
    if agent.risk_score >= 7.0:
        return 2
    return 3

def access_rank(level: int) -> int:
    # Más barrera ⇒ más prioridad (3 -> 1, 2 -> 2, 1 -> 3)
    return 4 - level

def priority_key(agent: Agent, waiting_days: float, dist_weight: float = 0.03) -> tuple:
    dist_term = -dist_weight * (agent.nearest_km or 0.0)
    return (
        tier(agent),
        -agent.risk_score,
        access_rank(agent.access_level),
        dist_term,
        -waiting_days,
        agent.id,
    )

def no_show_prob(agent: Agent) -> float:
    base = 0.05 + 0.05*(agent.access_level - 1)     # 0.05/0.10/0.15
    adj = base - 0.02*agent.reminders_sent + 0.5*agent.reticence
    return float(min(0.9, max(0.0, adj)))


def sample_agents(N: int) -> List[Agent]:
    agents = []
    subpops = np.random.choice(['A', 'B', 'C'], size=N, p=[0.35, 0.25, 0.40])

    for i in range(N):
        sp = subpops[i]
        if sp == 'A':  # Mayores / clínico alto
            age = int(np.clip(np.random.normal(68, 8), 50, 95))
            risk = float(np.clip(np.random.normal(8.5, 1.2), 0, 10))
            role = 'other'
            p_access = np.array([0.25, 0.45, 0.30])
        elif sp == 'B':  # Sanitario
            age = int(np.clip(np.random.normal(40, 9), 22, 75))
            risk = float(np.clip(np.random.normal(6.5, 1.5), 0, 10))
            role = np.random.choice(['healthcare', 'other'], p=[0.85, 0.15])
            p_access = np.array([0.55, 0.35, 0.10])
        else:  # 'C' Docentes / comunidad educativa
            age = int(np.clip(np.random.normal(38, 10), 22, 75))
            risk = float(np.clip(np.random.normal(5.5, 1.2), 0, 10))
            role = np.random.choice(['teacher', 'other'], p=[0.7, 0.3])
            p_access = np.array([0.45, 0.40, 0.15])

        access = int(np.random.choice([1, 2, 3], p=p_access))
        x, y = float(np.random.uniform(0, 50)), float(np.random.uniform(0, 50))
        ret = float(np.random.beta(2.5, 5.0))
        agents.append(Agent(i, age, role, sp, risk, access, x, y, ret))

    # Garantizar ≥20% de access=3 por subpoblación 
    for sp in ['A', 'B', 'C']:
        idx = [a for a in agents if a.subpop == sp]
        low = [a for a in idx if a.access_level == 3]
        share = len(low)/max(1, len(idx))
        if share < 0.20:
            deficit = int(round(0.20*len(idx))) - len(low)
            candidates = [a for a in idx if a.access_level == 2]
            for a in candidates[:deficit]:
                a.access_level = 3

    return agents

def sample_clinics(K: int) -> List[Clinic]:
    clinics = []
    for k in range(K):
        x, y = float(np.random.uniform(5, 45)), float(np.random.uniform(5, 45))
        mean_slots = int(np.random.randint(35, 60))  # slots/día
        clinics.append(Clinic(k, x, y, mean_slots, reserved_low_access_frac=0.25))
    return clinics

class VaccinationSim:
    def __init__(self, agents: List[Agent], clinics: List[Clinic], T_days: int = 20, max_km: float = 15.0):
        self.agents = agents
        self.clinics = clinics
        self.T_days = T_days
        self.max_km = max_km
        self.t = 0.0
        self.serial = 0
        self.EQ = []  # event queue (heap por tiempo)
        self.PQ = []  # priority queue de elegibles
        self.log = []

        # Precompute reachability y distancia al más cercano
        clinic_xy = [(c.x, c.y) for c in self.clinics]
        for a in self.agents:
            dists = [distance((a.x, a.y), xy) for xy in clinic_xy] if clinic_xy else [float("inf")]
            a.nearest_km = min(dists)
            a.reachable = {c.id for c in self.clinics if distance((a.x, a.y), (c.x, c.y)) <= self.max_km}

    # Infra de cola de eventos / logging
    def push_event(self, time: float, etype: VEvent, payload: dict):
        self.serial += 1
        heapq.heappush(self.EQ, (time, self.serial, etype, payload))

    def log_event(self, ts: float, kind: str, agent: Optional[Agent] = None, clinic_id: int = -1,
                  status: Optional[str] = None, dose: int = 1):
        self.log.append({
            "ts": ts,
            "event": kind,
            "agent_id": None if agent is None else agent.id,
            "age": None if agent is None else agent.age,
            "role": None if agent is None else agent.role,
            "subpop": None if agent is None else agent.subpop,
            "risk_score": None if agent is None else agent.risk_score,
            "access_level": None if agent is None else agent.access_level,
            "clinic_id": clinic_id,
            "dose_num": dose,
            "status": status,
        })

    # Elegibilidad temporal
    def eligibility_day(self, a: Agent) -> int:
        t = tier(a)
        if t == 1: return 0
        if t == 2: return 5
        return 10

    def build(self):
        for a in self.agents:
            te = self.eligibility_day(a)
            self.push_event(te + 0.2, VEvent.ELIGIBLE, {"aid": a.id})

        # Cupos por clínica y día (slots ~ Poisson)
        for d in range(self.T_days):
            for c in self.clinics:
                slots = np.random.poisson(max(1, c.mean_daily_slots))
                rsv = int(round(c.reserved_low_access_frac * slots)) 
                for s in range(slots):
                    payload = {"clinic_id": c.id, "reserved_for": 3 if s < rsv else None}
                    t = d + 0.4 + 0.5*np.random.rand()
                    self.push_event(t, VEvent.SLOT_OPEN, payload)

    def add_to_priority_queue(self, a: Agent, now: float):
        a.ts_eligible = now if a.ts_eligible is None else a.ts_eligible
        key = priority_key(a, waiting_days=now - a.ts_eligible)
        self.serial += 1
        heapq.heappush(self.PQ, PQItem(key, self.serial, a.id))

    def handle_slot(self, now: float, clinic_id: int, reserved_for: Optional[int], allow_fallback: bool = True):
        clinic = self.clinics[clinic_id]
        buffer = []
        picked = None

        # Recorre toda la PQ si hace falta (evita perder cupos por "top 40")
        while self.PQ:
            item = heapq.heappop(self.PQ)
            a = self.agents[item.aid]
            if a.vaccinated:
                continue
            if reserved_for is not None and a.access_level != reserved_for:
                buffer.append(item); continue
            if clinic_id not in a.reachable:
                buffer.append(item); continue
            picked = a
            break

        # devolver elementos no usados al heap
        for it in buffer:
            heapq.heappush(self.PQ, it)

        if picked is None:
            if reserved_for is not None and allow_fallback:
                self.log_event(now, "SLOT_OPEN_UNUSED", None, clinic_id, status="unused_reserved")
                self.handle_slot(now, clinic_id, reserved_for=None, allow_fallback=False)
                return
            self.log_event(now, "SLOT_OPEN_UNUSED", None, clinic_id, status="unused")
            return

        # Intento de vacunación con reintentos ante no-show
        tries = 0
        while tries < 5 and picked is not None:
            tries += 1
            if random.random() < no_show_prob(picked):
                picked.reminders_sent += 1
                self.log_event(now, "NO_SHOW", picked, clinic_id, status="no_show")
                self.add_to_priority_queue(picked, now)
                # Buscar siguiente rápido
                picked = None
                temp_buffer = []
                while self.PQ:
                    it = heapq.heappop(self.PQ)
                    a2 = self.agents[it.aid]
                    if a2.vaccinated: continue
                    if reserved_for is not None and a2.access_level != reserved_for:
                        temp_buffer.append(it); continue
                    if clinic_id not in a2.reachable:
                        temp_buffer.append(it); continue
                    picked = a2; break
                for it in temp_buffer:
                    heapq.heappush(self.PQ, it)
            else:
                picked.vaccinated = True
                picked.dose_num += 1
                self.log_event(now, "VACCINATE", picked, clinic_id, status="administered")
                return

        if picked is None:
            self.log_event(now, "SLOT_OPEN_UNUSED", None, clinic_id, status="unused")

    def run(self) -> pd.DataFrame:
        self.build()
        while self.EQ:
            t, _, etype, payload = heapq.heappop(self.EQ)
            self.t = t
            if etype == VEvent.ELIGIBLE:
                a = self.agents[payload["aid"]]
                self.add_to_priority_queue(a, t)
                self.log_event(t, "ELIGIBLE", a, status="eligible")
            elif etype == VEvent.SLOT_OPEN:
                self.log_event(t, "SLOT_OPEN", None, payload["clinic_id"], status=f"open_q={len(self.PQ)}")
                self.handle_slot(t, payload["clinic_id"], payload["reserved_for"])
        return pd.DataFrame(self.log)


def lorenz_and_gini(pop_by_group: pd.Series, dose_by_group: pd.Series, title: str, out_png: str):
    pop = pop_by_group.reindex(pop_by_group.index).fillna(0).astype(float)
    dose = dose_by_group.reindex(pop_by_group.index).fillna(0).astype(float)

    cum_pop = (pop / pop.sum()).cumsum()
    cum_dose = (dose / max(1.0, dose.sum())).cumsum()

    x = np.array([0.0] + cum_pop.tolist())
    y = np.array([0.0] + cum_dose.tolist())

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("Población acumulada")
    plt.ylabel("Dosis acumuladas")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    try:
        area_under = np.trapezoid(y, x)   # numpy >= 1.20
    except Exception:
        area_under = np.trapz(y, x)       # fallback
    gini = 1 - 2*area_under
    return gini

def heatmap_wait(w: pd.DataFrame, out_png: str):
    heat = w.groupby(["subpop", "access_level"])["wait"].mean().unstack().reindex(index=["A", "B", "C"], columns=[1, 2, 3])
    plt.figure(figsize=(6, 4))
    plt.imshow(heat.values, aspect="auto")
    plt.xticks(ticks=[0, 1, 2], labels=["Acceso 1", "Acceso 2", "Acceso 3"])
    plt.yticks(ticks=[0, 1, 2], labels=["SP-A", "SP-B", "SP-C"])
    plt.title("Espera media a vacunación (días)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


if __name__ == "__main__":
    N_AGENTS = 2000
    N_CLINICS = 5
    T_DAYS = 20

    agents = sample_agents(N_AGENTS)
    clinics = sample_clinics(N_CLINICS)

    sim = VaccinationSim(agents, clinics, T_DAYS, max_km=15.0)
    log_df = sim.run()

    log_df.to_csv("vax_log.csv", index=False)

    vacc = log_df[log_df.event == "VACCINATE"].copy()
    vacc["day"] = vacc["ts"].astype(float).apply(np.floor).astype(int)
    series = vacc.groupby("day").size()

    plt.figure(figsize=(8, 4))
    if len(series) <= 1:
        x = series.index.values if len(series) else np.array([0])
        y = series.values if len(series) else np.array([0])
        plt.stem(x, y)  
    else:
        plt.plot(series.index.values, series.values, marker="o")
    plt.title("Dosis administradas por día")
    plt.xlabel("Día")
    plt.ylabel("Dosis")
    plt.tight_layout()
    plt.savefig("series_dosis.png", dpi=160)
    plt.close()

    pop_by_access = pd.Series(
        [sum(a.access_level == lvl for a in agents) for lvl in [3, 2, 1]],
        index=[3, 2, 1],
        dtype=float
    )
    dose_by_access = vacc.groupby("access_level").size().reindex([3, 2, 1]).fillna(0).astype(float)
    gini = lorenz_and_gini(pop_by_access, dose_by_access,
                           "Curva de Lorenz — Dosis por nivel de acceso (3 primero)",
                           "lorenz_access.png")

    elig = log_df[log_df.event == "ELIGIBLE"][["agent_id", "ts"]].rename(columns={"ts": "ts_elig"})
    admin = log_df[(log_df.event == "VACCINATE") & (log_df.status == "administered")][["agent_id", "ts"]].rename(columns={"ts": "ts_vax"})
    w = pd.merge(elig, admin, on="agent_id", how="inner")
    meta = pd.DataFrame([{"agent_id": a.id, "subpop": a.subpop, "access_level": a.access_level} for a in agents])
    w = w.merge(meta, on="agent_id", how="left")
    w["wait"] = w["ts_vax"] - w["ts_elig"]
    heatmap_wait(w, "heatmap_wait.png")

    summary = vacc.groupby(["subpop", "access_level"]).size().unstack().fillna(0).astype(int).reindex(index=["A", "B", "C"], columns=[1, 2, 3])
    summary["Total"] = summary.sum(axis=1)
    summary.to_csv("resumen_subpop_access.csv")


    print("=== RESUMEN ===")
    print(f"N agentes: {len(agents)}")
    print(f"N clínicas: {len(clinics)}")
    print(f"Días simulados: {T_DAYS}")
    print(f"Dosis totales: {len(vacc)}")
    print(f"Gini (dosis vs población por acceso): {gini:.3f}")
    print("\nCobertura por subpoblación × acceso:")
    print(summary)
    print("\nArchivos generados:")
    print(" - vax_log.csv")
    print(" - series_dosis.png")
    print(" - lorenz_access.png")
    print(" - heatmap_wait.png")
    print(" - resumen_subpop_access.csv")