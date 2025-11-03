# Meeting Minutes

---

## 2025-08-26

### Attendees
- Shreyas
- David
- Francesca
- Alexander

### Agenda
- Thesis logistics
- Literature review overview
- NTK problem setup
- Scheduling future meetings

### Discussion & Decisions
- **Thesis logistics**: You will check the rules on who can serve as an external examiner on the defense panel.
- **Literature review**: Decided to start by formalizing the NTK problem in a simpler case before moving to more complex settings.
- **NTK experiments**: Plan to try out some initial coding experiments with NTK and document the results.
- **Future meetings**: You should set up a weekly meeting schedule with both your advisor and supervisor.

### Action Items
- [ ] Find out eligibility rules for external examiners.
- [ ] Draft a simplified formalization of the NTK problem.
- [ ] Write and document initial NTK code experiments.
- [ ] Propose a recurring weekly meeting time with advisor and supervisor.

---

## 2025-10-09

### Attendees
- Shreyas
- Francesca
- Alexander

### Agenda
- Go over the thesis proposal, feedback on the approach and tentative plan
- Discuss goals and set expectations for the first stage review
- Discuss communication channels and schedule for weekly meetings

### Discussion & Decisions
- Hanin-Nica is from 2019, comparatively old research with seemlingly less follow-up.
- Go through more literature to find references for weak learning or look at more approaches.

### Action Items
- [ ] Set up a regular meeting with Francesca and Alexander after figuring out a common time.
- [ ] Implement Hanin-Nica baseline to verify the existence of weak learning.
---

## 2025-10-13

### Attendees
- Shreyas
- David

### Agenda
- Discuss communication channel
- Give a brief of the previous meeting
- Go over the initial plan for baseline testing

### Discussion & Decisions
- Briefly went over NTK theory and discussed the claims from Hanin-Nica

### Action Items
- [ ] Implement Hanin-Nica baseline to verify the existence of weak learning.

## 2025-10-20

### Attendees
- Shreyas
- David

### Agenda
- Discuss the initial NTK experiment 

### Discussions & Decisions
- Discussed the preliminary results, agreement on the experiments to run

### Action Items
- [ ] Create an experiment log for logging all results.

## 2025-10-27

### Attendees
- Shreyas
- David

### Agenda
- Status of current experiment
    - Kernel-profile replication done (variance $\downarrow$ with width).
    - Function-space experiments on \(S^1\): simple vs complex targets; width×seed sweeps; RelErr computed.
- Discuss experiment for Hanin-Nica weak-learning regime.

### Discussions & Decisions
- Discussions about behaviour of the kernel profile, why does it peak at small widths? 
    - Possible reasons could be that the gradients form a wedge / cone-like slice on the unit sphere in the weight space.
- The difference in the relative errors for complex regression task didn't seem significant in absolute values, despite the decreasing trend.
- All the widths seem to be very similar, meaning 512 width network seems to perform just as well as 16,000 width network.
- Discussed reducing the harmonics in the regression task to make it simpler. 
- Try to understand this convergence in infinite vs finite before moving onto Hanin-Nica. 

### Action Items
- [ ] Try to make the task simpler to see better convergence with increasing widths
- [ ] Compare convergence in the lower frequncies vs high frequencies.

## 2025-11-11

### Attendees
- Shreyas
- David
- Francesca

### Agenda
- Discuss the possibility of a monthly sync with all supervisors together
- Brief update of the experiments conducted and determine next steps

### Discussions & Decisions

### Action Items
