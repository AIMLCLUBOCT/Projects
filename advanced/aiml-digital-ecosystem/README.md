# 🌐 AIML Club OCT Digital Ecosystem & Voice Portal

> Scalable cloud and web architecture powering the digital infrastructure of **AIML Club – Oriental College of Technology, Bhopal**.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg?style=flat-square)](https://aimlcluboct.in)
[![Architect: Umesh Patel](https://img.shields.io/badge/Architect-Umesh_Patel-blue.svg?style=flat-square)](https://github.com/UmeshCode1)
[![Stack: Next.js + Vercel](https://img.shields.io/badge/Stack-Next.js_|_React_|_Tailwind-purple.svg?style=flat-square)](https://nextjs.org/)

---

## 📌 Architecture Overview

The digital presence of AIML Club OCT is designed as a distributed, high-availability ecosystem comprising specialized subdomains and service portals:

```mermaid
flowchart TD
    User([Club Members & Students]) --> DNS{Cloudflare / DNS}
    DNS --> Main[aimlcluboct.in<br/>Main Portal & Team Directory]
    DNS --> Social[social.aimlcluboct.in<br/>Linktree & Digital Social Hub]
    DNS --> Live[live.aimlcluboct.in<br/>Real-Time Broadcasts & Events]
    DNS --> Voice[voice.aimlcluboct.in<br/>Student Feedback & Idea Box]
    Main --> Blog[aimlcluboct.in/blog<br/>Technical Articles & Guides]
    Main --> Team[aimlcluboct.in/team<br/>Leadership & Coordinators]
```

---

## 🚀 Ecosystem Portals

| Subdomain | Function | Tech Stack | Status |
| :--- | :--- | :--- | :---: |
| [**aimlcluboct.in**](https://aimlcluboct.in) | Primary website: Club overview, team catalog, workshops, resources | Next.js, React, Tailwind CSS | 🟢 Live |
| [**social.aimlcluboct.in**](https://social.aimlcluboct.in) | Consolidated link hub: Social networks, community links | Modern Responsive Web | 🟢 Live |
| [**live.aimlcluboct.in**](https://live.aimlcluboct.in) | Real-time event broadcasting and live results | Real-time dashboard | 🟢 Live |
| [**voice.aimlcluboct.in**](https://voice.aimlcluboct.in) | Interactive suggestions and anonymous feedback portal | Form API + Database | 🟢 Live |

---

## 👥 Lead Architect
- **Umesh Patel** ([@UmeshCode1](https://github.com/UmeshCode1)) — Vice President & System Architect, AIML Club OCT.
