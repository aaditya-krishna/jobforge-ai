# Relevance labeling guide

`labels.csv` marks each (resume, company, title) in a resume's candidate pool as relevant (1) or not (0).
A job is **relevant** if a recruiter would reasonably send it to this candidate: they could credibly apply
and would plausibly want to.

**Who labeled:** every label in `labels.csv` was assigned by Claude (the AI assistant that built the
project) from each posting's title, company and the opening of its description, applying the rules below.
No person has reviewed them yet. The resumes are synthetic. Treat the precision numbers as a
consistency check of the ranking, not as ground truth; relabeling by a person is the main open item.

## General rules (all resumes)

1. **Role family must match** the resume's core work (table below). Adjacent roles count only if most of
   the day-to-day work is the same.
2. **Level must be plausible.** Every resume has 3–7 years of experience, so:
   - not relevant: internships, new-grad / university-grad only roles, and "Distinguished" / "Fellow" roles;
   - not relevant for individual contributors: people-manager roles (Manager, Director, Head of, VP),
     unless the resume shows people management;
   - Principal roles count only for resumes with 6+ years.
3. **Location and company are ignored**: the resumes don't state location preferences.
4. Unclear cases use the title first, then the description opening.

## Role families

| Resume | Relevant | Not relevant (examples) |
|---|---|---|
| backend_engineer (5y) | backend / platform / distributed-systems software engineering, payments/infra backend | frontend-only, mobile, ML research, data science, SRE-only on-call ops, sales engineering |
| ml_engineer (4y) | ML engineering, applied ML, recommendation/ranking/personalization, ML platform | pure research scientist roles requiring a PhD track record, data analyst, non-ML backend |
| data_engineer (6y) | data engineering, data platform, analytics engineering, data infrastructure | data science/analytics-only, ML research, generic backend without data focus |
| frontend_engineer (4y) | frontend / web / UI engineering, full-stack with a frontend focus, design-systems engineering | backend-only, mobile-native, product design (non-engineering) |
| ios_engineer (5y) | iOS engineering, mobile engineering that includes iOS | Android-only, web frontend, backend |
| sre_platform (7y) | SRE, platform/infrastructure engineering, DevOps, production engineering, cloud infrastructure | security-only, backend product engineering, IT helpdesk |
| security_engineer (5y) | security engineering: detection & response, cloud/app/product/infrastructure security, security operations | GRC-only/policy, physical security, IT, non-security SRE |
| data_analyst (3y) | data/business/product analyst, analytics roles centered on SQL/BI | data engineering, data science requiring ML modeling, finance accounting roles |
| robotics_perception (5y) | perception, computer vision, sensor fusion, autonomy/robotics software, state estimation | hardware/mechanical, generic backend, ML research unrelated to robotics |
| embedded_firmware (6y) | embedded / firmware software, low-level systems on hardware, controls software | hardware design without software, cloud backend, test technician |
| account_executive (7y) | account executive / enterprise sales / account management (quota-carrying) | sales engineering/solutions architect, SDR/BDR (more junior), sales leadership with direct reports, marketing |
| product_designer (5y) | product / UX / interaction design | design engineering (code-heavy), brand/marketing design, user research-only roles, design management |
