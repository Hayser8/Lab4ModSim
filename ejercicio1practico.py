from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd
import heapq, math, random
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

class State(Enum):
    S = 0
    E = 1
    I = 2
    R = 3

@dataclass
class Agent:
    id: int
    ward: int
    role: str
    state: State = State.S
    susceptibility: float = 1.0
    infectivity: float = 1.0
    days_incub_left: int = 0
    days_infec_left: int = 0
    t_E_to_I: float | None = None
    t_I_to_R: float | None = None
    infector_id: int | None = None
    secondary_infections: int = 0

def avg_contacts(contacts_per_day, staff_frac, staff_mult):
    return contacts_per_day * ((1 - staff_frac) + staff_frac * staff_mult)

def true_Reff(agents, init_I):
    total_sec = sum(a.secondary_infections for a in agents)
    total_infected = sum(1 for a in agents if a.state != State.S)
    primaries = max(1, total_infected - init_I)
    return total_sec / primaries

class HospitalSimDiscrete:
    def __init__(
        self,
        N=300,
        W=5,
        init_I=5,
        days=60,
        beta=0.06,
        contacts_per_day=8,
        inc_mean_days=3.0,
        inf_mean_days=5.0,
        staff_frac=0.25,
        staff_contact_mult=1.5
    ):
        self.N, self.W, self.days = N, W, days
        self.beta = beta
        self.contacts_per_day = contacts_per_day
        self.inc_mean_days = inc_mean_days
        self.inf_mean_days = inf_mean_days
        self.staff_frac = staff_frac
        self.staff_contact_mult = staff_contact_mult
        self.init_I = init_I
        self.agents: list[Agent] = []
        for i in range(N):
            role = "staff" if np.random.rand() < staff_frac else "patient"
            ward = np.random.randint(0, W)
            a = Agent(id=i, ward=ward, role=role)
            self.agents.append(a)
        init_ids = np.random.choice(N, size=init_I, replace=False)
        for idx in init_ids:
            a = self.agents[idx]
            a.state = State.I
            a.days_infec_left = self._sample_days(self.inf_mean_days)
        self.daily_incidence = np.zeros(days, dtype=int)
        self.daily_prevalence = np.zeros(days, dtype=int)
        self.daily_I_by_ward = np.zeros((days, W), dtype=int)

    @staticmethod
    def _sample_days(mean):
        return max(1, int(np.random.exponential(mean) + 0.5))

    def _ward_lists(self):
        wards = [[] for _ in range(self.W)]
        for a in self.agents:
            wards[a.ward].append(a.id)
        return wards

    def run(self):
        for t in range(self.days):
            for a in self.agents:
                if a.state == State.E:
                    a.days_incub_left -= 1
                    if a.days_incub_left <= 0:
                        a.state = State.I
                        a.days_infec_left = self._sample_days(self.inf_mean_days)
                elif a.state == State.I:
                    a.days_infec_left -= 1
                    if a.days_infec_left <= 0:
                        a.state = State.R
            wards = self._ward_lists()
            I_counts = np.zeros(self.W, dtype=int)
            N_counts = np.zeros(self.W, dtype=int)
            for w in range(self.W):
                ids = wards[w]
                if not ids:
                    continue
                N_counts[w] = len(ids)
                I_counts[w] = sum(1 for i in ids if self.agents[i].state == State.I)
            new_exposures = []
            for w in range(self.W):
                if N_counts[w] == 0:
                    continue
                Iw = I_counts[w]
                if Iw == 0:
                    continue
                for i in wards[w]:
                    a = self.agents[i]
                    if a.state != State.S:
                        continue
                    cp = self.contacts_per_day * (self.staff_contact_mult if a.role == "staff" else 1.0)
                    lam = self.beta * (Iw / N_counts[w]) * cp * a.susceptibility
                    p_inf = 1.0 - math.exp(-lam)
                    if np.random.rand() < p_inf:
                        a.state = State.E
                        a.days_incub_left = self._sample_days(self.inc_mean_days)
                        new_exposures.append(a.id)
                        infector_idx = np.random.choice([j for j in wards[w] if self.agents[j].state == State.I])
                        self.agents[i].infector_id = infector_idx
                        self.agents[infector_idx].secondary_infections += 1
            self.daily_incidence[t] = len(new_exposures)
            self.daily_prevalence[t] = sum(1 for a in self.agents if a.state == State.I)
            self.daily_I_by_ward[t, :] = I_counts
        total_infected = sum(1 for a in self.agents if a.state != State.S)
        attack_rate = total_infected / self.N
        R_eff = true_Reff(self.agents, self.init_I)
        return {
            "daily_incidence": pd.Series(self.daily_incidence, name="incidence"),
            "daily_prevalence": pd.Series(self.daily_prevalence, name="prevalence"),
            "daily_I_by_ward": pd.DataFrame(self.daily_I_by_ward, columns=[f"ward_{w}" for w in range(self.W)]),
            "attack_rate": attack_rate,
            "R_eff": R_eff,
        }

class EventType(Enum):
    EXPOSE = auto()
    E_TO_I = auto()
    I_TO_R = auto()
    CONTACT = auto()
    TRANSFER = auto()
    SAMPLER = auto()

@dataclass(order=True)
class Event:
    time: float
    order: int
    etype: EventType = field(compare=False)
    aid: int = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)

class HospitalSimEvent:
    def __init__(
        self,
        N=300,
        W=5,
        init_I=5,
        T=60.0,
        contact_rate=8.0,
        p_transmit=0.06,
        inc_rate=1/3.0,
        rec_rate=1/5.0,
        transfer_rate=0.02,
        staff_frac=0.25,
        staff_contact_mult=1.5
    ):
        self.N, self.W, self.T = N, W, T
        self.contact_rate = contact_rate
        self.p_transmit = p_transmit
        self.inc_rate = inc_rate
        self.rec_rate = rec_rate
        self.transfer_rate = transfer_rate
        self.staff_frac = staff_frac
        self.staff_contact_mult = staff_contact_mult
        self.init_I = init_I
        self.t = 0.0
        self.counter = 0
        self.Q: list[Event] = []
        self.agents: list[Agent] = []
        for i in range(N):
            role = "staff" if np.random.rand() < staff_frac else "patient"
            ward = np.random.randint(0, W)
            a = Agent(id=i, ward=ward, role=role)
            self.agents.append(a)
        init_ids = np.random.choice(N, size=init_I, replace=False)
        for idx in init_ids:
            self._set_I(idx)
            self._schedule_contact(idx)
            self._schedule_I_to_R(idx)
        self.expose_times = []
        self.prevalence_samples = []
        self.I_by_ward_samples = defaultdict(lambda: [])
        self._schedule_day_sampler()
        for i in range(N):
            self._schedule_transfer(i)

    def _push(self, time: float, etype: EventType, aid: int, payload=None):
        if payload is None:
            payload = {}
        self.counter += 1
        heapq.heappush(self.Q, Event(time, self.counter, etype, aid, payload))

    def _exp(self, rate):
        return np.random.exponential(1.0 / rate)

    def _ward_members(self, w):
        return [a.id for a in self.agents if a.ward == w]

    def _set_S(self, i): self.agents[i].state = State.S
    def _set_E(self, i): self.agents[i].state = State.E
    def _set_I(self, i): self.agents[i].state = State.I
    def _set_R(self, i): self.agents[i].state = State.R

    def _schedule_contact(self, i):
        a = self.agents[i]
        if a.state != State.I: return
        rate = self.contact_rate * (self.staff_contact_mult if a.role == "staff" else 1.0) * a.infectivity
        dt = self._exp(rate)
        self._push(self.t + dt, EventType.CONTACT, i)

    def _schedule_E_to_I(self, i):
        dt = self._exp(self.inc_rate)
        self.agents[i].t_E_to_I = self.t + dt
        self._push(self.agents[i].t_E_to_I, EventType.E_TO_I, i)

    def _schedule_I_to_R(self, i):
        dt = self._exp(self.rec_rate)
        self.agents[i].t_I_to_R = self.t + dt
        self._push(self.agents[i].t_I_to_R, EventType.I_TO_R, i)

    def _schedule_transfer(self, i):
        dt = self._exp(self.transfer_rate)
        self._push(self.t + dt, EventType.TRANSFER, i)

    def _schedule_day_sampler(self):
        next_day = math.ceil(self.t)
        while next_day <= math.floor(self.T):
            self._push(next_day, EventType.SAMPLER, -1)
            next_day += 1

    def _handle_contact(self, e: Event):
        inf = self.agents[e.aid]
        if inf.state != State.I:
            return
        ward_ids = self._ward_members(inf.ward)
        if len(ward_ids) <= 1:
            self._schedule_contact(inf.id)
            return
        target_id = random.choice(ward_ids)
        if target_id == inf.id:
            self._schedule_contact(inf.id)
            return
        tgt = self.agents[target_id]
        if tgt.state == State.S and random.random() < (self.p_transmit * tgt.susceptibility):
            self._push(self.t, EventType.EXPOSE, target_id, {"infector": inf.id})
        self._schedule_contact(inf.id)

    def _handle_expose(self, e: Event):
        a = self.agents[e.aid]
        if a.state != State.S:
            return
        a.state = State.E
        a.infector_id = e.payload.get("infector")
        if a.infector_id is not None:
            self.agents[a.infector_id].secondary_infections += 1
        self.expose_times.append(self.t)
        self._schedule_E_to_I(a.id)

    def _handle_E_to_I(self, e: Event):
        a = self.agents[e.aid]
        if a.state != State.E:
            return
        a.state = State.I
        self._schedule_I_to_R(a.id)
        self._schedule_contact(a.id)

    def _handle_I_to_R(self, e: Event):
        a = self.agents[e.aid]
        if a.state != State.I:
            return
        a.state = State.R
        a.t_I_to_R = None

    def _handle_transfer(self, e: Event):
        a = self.agents[e.aid]
        if self.W > 1:
            new_w = random.randrange(self.W)
            a.ward = new_w
        self._schedule_transfer(a.id)

    def _sample_day(self, day_int: int):
        I_count = sum(1 for a in self.agents if a.state == State.I)
        self.prevalence_samples.append((day_int, I_count))
        counts = defaultdict(int)
        for a in self.agents:
            if a.state == State.I:
                counts[a.ward] += 1
        self.I_by_ward_samples[day_int] = [counts[w] for w in range(self.W)]

    def run(self):
        while self.Q and self.t <= self.T:
            e = heapq.heappop(self.Q)
            self.t = e.time
            if self.t > self.T:
                break
            if e.etype == EventType.CONTACT:
                self._handle_contact(e)
            elif e.etype == EventType.EXPOSE:
                self._handle_expose(e)
            elif e.etype == EventType.E_TO_I:
                self._handle_E_to_I(e)
            elif e.etype == EventType.I_TO_R:
                self._handle_I_to_R(e)
            elif e.etype == EventType.TRANSFER:
                self._handle_transfer(e)
            elif e.etype == EventType.SAMPLER:
                self._sample_day(int(e.time))
        if self.expose_times:
            max_day = int(math.floor(max(self.expose_times)))
        else:
            max_day = int(math.floor(self.T))
        bins = np.arange(0, max_day + 2)
        hist, _ = np.histogram(self.expose_times, bins=bins)
        daily_incidence = pd.Series(hist, index=np.arange(0, max_day + 1), name="incidence")
        if self.prevalence_samples:
            days, vals = zip(*self.prevalence_samples)
            daily_prevalence = pd.Series(vals, index=days, name="prevalence").sort_index()
        else:
            daily_prevalence = pd.Series(dtype=int, name="prevalence")
        if self.I_by_ward_samples:
            max_key = max(self.I_by_ward_samples.keys())
            mat = np.zeros((max_key + 1, self.W), dtype=int)
            for d, vec in self.I_by_ward_samples.items():
                mat[d, :] = vec
            daily_I_by_ward = pd.DataFrame(mat, columns=[f"ward_{w}" for w in range(self.W)])
        else:
            daily_I_by_ward = pd.DataFrame(columns=[f"ward_{w}" for w in range(self.W)])
        total_infected = sum(1 for a in self.agents if a.state != State.S)
        attack_rate = total_infected / self.N
        R_eff = true_Reff(self.agents, self.init_I)
        return {
            "daily_incidence": daily_incidence,
            "daily_prevalence": daily_prevalence,
            "daily_I_by_ward": daily_I_by_ward,
            "attack_rate": attack_rate,
            "R_eff": R_eff,
        }

if __name__ == "__main__":
    disc = HospitalSimDiscrete(
        N=300, W=4, init_I=6, days=60,
        beta=0.06, contacts_per_day=8,
        inc_mean_days=3.0, inf_mean_days=5.0,
        staff_frac=0.3, staff_contact_mult=1.6
    )
    out_d = disc.run()
    print("=== Discreto (diario) ===")
    print("Attack rate:", round(out_d["attack_rate"], 3))
    print("R_eff (true por caso):", round(out_d["R_eff"], 3))
    print(out_d["daily_incidence"].head())

    p_match = 0.06
    ev = HospitalSimEvent(
        N=300, W=4, init_I=6, T=60.0,
        contact_rate=8.0, p_transmit=p_match,
        inc_rate=1/3.0, rec_rate=1/5.0,
        transfer_rate=0.02, staff_frac=0.3, staff_contact_mult=1.6
    )
    out_e = ev.run()
    print("\n=== Evento (continuo) ===")
    print("Attack rate:", round(out_e["attack_rate"], 3))
    print("R_eff (true por caso):", round(out_e["R_eff"], 3))
    print(out_e["daily_incidence"].head())

    plt.figure()
    out_d["daily_incidence"].plot()
    plt.title("Incidencia diaria (Discreto)")
    plt.xlabel("Día")
    plt.ylabel("Casos nuevos")
    plt.show()

    plt.figure()
    out_e["daily_incidence"].plot()
    plt.title("Incidencia diaria (Eventos)")
    plt.xlabel("Día")
    plt.ylabel("Casos nuevos")
    plt.show()
