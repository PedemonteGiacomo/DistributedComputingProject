# Research Report: Peer-to-Peer System Analysis and Optimization

## Introduction
This report provides a detailed analysis of our work on the DC assignment, conducted by the group members Ali Haider and Giacomo Pedemonte, dated 23/02/2023. The research focused on simulations and theoretical expectations related to the DC assignment exercises.

## Objectives
1. **M/M/1 to M/M/N Queue Analysis:**
   - Implemented Supermarket Theorem for theoretical validation.
   - Explored variations in graphs based on lambda and supermarket choices.
2. **Storage Simulations:**
   - Conducted simulations for client-server and peer-to-peer configurations.
   - Analyzed block storage and retrieval strategies.

## M/M/1 and M/M/N Queue Analysis
- **M/M/1 Queue:** Simulated a FIFO queue, validating against theoretical expectations. Results showed a direct correlation between lambda and average time spent in the system.
- **M/M/N Queue:** Implemented supermarket model for queue selection. Explored the impact of different choices on system efficiency. Results aligned with theoretical predictions, especially with the use of two choices.

## Storage Simulations: Client-Server and Peer-to-Peer
- **Client-Server Model:** Explored various client-server configurations, optimizing the balance between clients and servers to prevent data loss.
- **Peer-to-Peer Model:** Conducted extensive simulations with different node configurations, focusing on block storage and recovery. Implemented Tit-for-Tat strategy with optimistic unchoke, significantly reducing data loss even in the presence of selfish nodes.

## Conclusion and Insights
- **Tit-for-Tat Strategy:** Implemented the Tit-for-Tat strategy with optimistic unchoke, enhancing cooperation among nodes and minimizing data loss.
- **Selfish Nodes Impact:** Explored the behavior of selfish nodes in the system. Identified optimal configurations that resist data loss even with a significant presence of selfish nodes.
- **Continuous Improvement:** Acknowledged the need for ongoing refinement to mitigate selfish behavior and enhance overall system robustness.

This report showcases our deep dive into queueing theory and storage system simulations. The findings underscore the importance of strategic queue selection and collaborative storage behaviors. Our work provides valuable insights into optimizing peer-to-peer systems, paving the way for future advancements in distributed computing paradigms.
