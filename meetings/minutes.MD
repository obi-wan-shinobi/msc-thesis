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
- [x] Find out eligibility rules for external examiners.
- [x] Draft a simplified formalization of the NTK problem.
- [x] Write and document initial NTK code experiments.
- [x] Propose a recurring weekly meeting time with advisor and supervisor.

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
- [x] Set up a regular meeting with Francesca and Alexander after figuring out a common time.
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
- [x] Implement Hanin-Nica baseline to verify the existence of weak learning.

## 2025-10-20

### Attendees
- Shreyas
- David

### Agenda
- Discuss the initial NTK experiment 

### Discussions & Decisions
- Discussed the preliminary results, agreement on the experiments to run

### Action Items
- [x] Create an experiment log for logging all results.

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

### Agenda
- Discuss the possibility of a monthly sync with all supervisors together (postponed for another time)
- Brief update of the experiments conducted and determine next steps

### Discussions & Decisions

- Very wide seems to converge, but does it converge for 10-100? Maybe with a good learning rate?
- Check how much the eigenvalues change during training for finite networks?
- Can we bound the learning rate using the eigen values of empirical NTK at timestep t.

### Action Items
- [x] Try to see is we can make 100 width network converge with different learning rates + more timesteps
- [x] Design an experiment to see how eigenvalues of the empirical NTK change

## 2025-11-18

### Attendees
- Shreyas
- Alexander

### Agenda
- Brief update about experiments performed and next steps
- NTK kernel profile reproduction to confirm deterministic behaviour for large widths
- Convergence of finite-width 1-hidden layer ReLU networks to analytic NTK predictions (infinite width)
- Discuss eigenvalue drift results

### Discussions & Decisions
- Try to understand what happens to the training dynamics when the eigenvalues stabilise.
- Following questions need to be addressed:
    - When the eigenvalues stop changing, are we in the NTK limit? 
    - The weights in the hidden layer cause non-convexity. If the eigenvalues stop, the empirical NTK stops changing, so does the "mesh" keep changing? 
    - What does this "stopping" imply for the training process? 
    - As long as the matrix changes, we are non-linear but once eigenvalues stabilise, is the model essentially linear?
    - Would it be possible to solve the problem using kernel regression once the eigenvalues stabalise?
- Look into "Conjugate gradient method" for estimates regarding gradient descent step size (learning rate).
- Question: in real world scenario, you don't know when your eigenvalues have converged... How do we determine this? Is it possible to understand what is the difference between regime befor and after eigenvalues stabalize

### Action items
- [x] Determine the point in training where the NTK matrix stops evolving and solve the system using kernel regression.
- [ ] Eigenvalues graph: Once the training stabalizes, assume that matrix doesn't change anymore, project residuals onto each eigenmode and check what happens.
- [ ] Take width 100 network, draw a mesh and determine if it is possible to approximate the function. 

## 2025-11-19

### Attendees
- Shreyas
- Francesca

### Agenda
- Brief update about experiments performed and next steps
- NTK kernel profile reproduction to confirm deterministic behaviour for large widths
- Convergence of finite-width 1-hidden layer ReLU networks to analytic NTK predictions (infinite width)
- Discuss eigenvalue drift results

### Discussions & Decisions
- Similar to previous meeting with Alexander.
- The widths of the networks are linked to the Fourier modes that the network can learn which are in-turn dependent on the eigenvalues.

### Action items
Same as before 

## 2025-11-20

### Attendees
- Shreyas 
- David

### Agenda
- Brief update about experiments with eigenvalue tracking and other meetings

### Discussions & Decisions
- Similar to previous meeting with Alexander
- We should check if this interpretation is still correct with another dataset which is different and more
complex.

### Action items
- [ ] Make a new dataset which is more complex to see if the interpretation is still correct 


## 2025-11-25

### Attendees
- Shreyas 
- Francesca

### Agenda
- Updates on NTK drift, freezing of NTK and lazy training regime

### Discussions & Decisions
- When the NTK freezes during training, the matrix becomes non-invertible because some eigenvalues
are very close to zero. We have to use regularized kernel regression to train with this kernel. 
- When you use regularisation, you move the function and this moving effect
causes you to worsen the approximation on high frequency peaks.
- For shallower networks, the solution obtained by kernel regression by taking the frozen kernel 
has lower error when compared to the solution obtained at the end of training the network. 
- Compare the error of the kernel regression vs final network predictions. Is it worth the computational cost? Compare the gain with the actual error.
- It would be interesting to characterize the transition from feature learning to lazy learning, maybe by finding a lower bound? 

### Action items
- [ ] Figure out a way to characterize the gain of training the network till the end vs doing kernel 
regression at the point where the empirical NTK freezes. 


## 2025-11-27

### Attendees
- Shreyas 
- David

### Agenda
- Updates on NTK drift, freezing of NTK and lazy training regime

### Discussions & Decisions
- The rate of convergence of NTK is inherently slow for large networks because it could be 
spending time on learning higher frequencies early on
- There is loss dip before the kernel freezes, is the loss dip significant?
- Do different eigenvalues converge at different rates for different widths?
- Maybe later, Vary input data once in a while, see if the eigenvalue
convergence depends on the amount of frequencies in the data. 
- The eigenvalues are linked to the data points so we need to renormalize. 
- Is it obvious that the largest eigenvalues will always converge to the 256? 
- Why is the kernel not stopping at the minimum loss? The learning rate for
this setting could be too large, try lowering it and retesting. 
- Gradients should be 0 at minimum loss, so why is the kernel changing? 
- David agrees with Francesca, try to figure out the gain of error vs computation. 
- Try to save the parameters at the frozen point and perform kernel regression
on different subsets. 

### Action items
- [ ] Lower the width to 45 or something and see if the eigenvalues still look the same
 
## 2025-12-04

### Attendees
- Shreyas
- Alexander 

### Agenda
- Updates about experiments so far, mainly the experiment with tracking NTK drift and lazy training.

### Discussions & Decisions
- The method of tracking whether the NTK is frozen is not entirely correct. It could be that the
magnitude from the initial state remains unchanged, imagine the NTK matrix is rotating on a circle
then in the norm, the distance from initial matrix will stop changing but the NTK isn't frozen. 
- Make sure to check data-dependent effects. It could be that the non-uniform distribution of
training points is causing the kernel regression solution to behave very badly on some regions.
- Does the max eigenvalue depend on the frequency of the problem or amount of data?
- Look into Gauss-Newton or Natural Gradient descent.
- Discussion on why the loss has saturated but kernel changes: maybe it is the learning rate since it is quite large...


## 2025-12-16

### Attendees
- Shreyas
- Alexander 
- Francesca

### Agenda
- Feedback on the interim-thesis manuscript plan. 
- Discuss related work and if it fits correctly into the plan.

### Discussions & Decisions
- Briefly explained the storyline; explain problems with deep learning theory.
- Introduce the limiting viewpoints that can be used by analysts: NTK, mean-field viewpoint, spectral bias
- Alexander: What is the fundamental difference between mean field view and NTK? And the assumptions?

## 2026-01-13

### Attendees
- Shreyas
- Alexander 

### Agenda
- Discussion on research question direction and some discussions about experiment 3.

### Discussions & Decisions
- Alexander: Check residual projections when eigen values stop changing, not before that.
- Suggestions on research question:

    - come up with one major question
    - explicitly formulate the question and put it in the document
    - Start your question and in the end in your thesis try to answer the questions

Discussion about convolution operator of NTK:
At finite width, it could be a discrete convolution?

Thesis committee form: 
- Antonis Papapantoleon
