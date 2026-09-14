# Problem Statement

## Background

The electricity grid is undergoing a rapid transition: as renewable generation (solar and wind) grows, grid operators must manage increasingly complex supply-demand dynamics. Unlike dispatchable thermal generation, renewables are weather-dependent and can underperform without warning. At the same time, demand is growing and becoming more volatile due to electrification of transport, heating, and industry.

## The Problem

Grid operators face five simultaneous challenges that today's tools handle in isolation:

1. **Demand Forecasting**: Predicting when load will spike is critical for securing reserve capacity, but forecast errors propagate into every downstream decision.

2. **Renewable Performance Monitoring**: Solar and wind assets underperform for many reasons — weather, equipment faults, curtailment, data errors. Without automated detection, operators discover problems too late.

3. **Root Cause Ambiguity**: When a solar plant generates less than expected, is it cloudy? Is there an inverter fault? Is the sensor reporting incorrectly? Misdiagnosis leads to unnecessary maintenance dispatches or, worse, unaddressed real faults.

4. **Supply-Demand Optimisation**: When a gap exists between supply and demand, operators must decide in real time whether to discharge BESS, activate demand response, call backup generation, or curtail renewables — each with different cost, speed, and environmental implications.

5. **Renewable Curtailment**: In surplus conditions, renewable energy is wasted unless BESS and flexible loads can absorb it. Curtailment decisions today are often reactive and non-optimal.

## Who Experiences This Problem

Grid control room operators at electricity transmission and distribution system operators (TSOs/DSOs) and renewable energy plant operators who must coordinate multiple information streams under time pressure.

## Why It Matters

- Unnecessary curtailment of renewable generation wastes clean energy and reduces project economics
- Undetected asset underperformance compounds over time into significant lost revenue
- Demand spikes that catch operators unprepared can require emergency generation at high cost
- The integration of AI assistants into control rooms requires tools that cite evidence, not guess

## What Was Missing

A single, integrated platform that chains demand forecasting → renewable performance monitoring → anomaly detection → root cause analysis → optimisation recommendation → operator briefing, with a conversational AI interface that uses computed results rather than fabricated answers.
