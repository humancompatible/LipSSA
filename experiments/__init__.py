"""Experiment infrastructure for the sampling estimators.

Every experiment is a grid of (network, method, seed) cells. Each cell runs
independently, writes one JSON record through `cache`, and is skipped on a
re-run once its record exists, so a grid can be spread over many short jobs and
resumed after an interruption. Notebooks only read the records and plot.
"""
