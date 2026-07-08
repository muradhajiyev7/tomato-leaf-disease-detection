# Project Overview

## Problem

Tomato leaf diseases can spread quickly and reduce crop yield when detected late. Manual inspection is time-consuming and may not scale well in greenhouse or field environments.

## Objective

The objective was to build an end-to-end deep learning system that classifies tomato leaf disease images and returns confidence-aware predictions suitable for prototype mobile testing.

## Target Users

- Farmers and greenhouse operators
- Agricultural students and researchers
- Applied AI teams exploring crop-health screening
- Mobile users capturing or uploading tomato leaf images

## Solution Summary

The system uses a PyTorch and EfficientNet-based computer vision pipeline. It combines preprocessing, disease classification, calibration, test-time augmentation, and confidence/margin-based accept/reject logic.

## Two-Stage Project Structure

Stage 1 focused on backbone experimentation and validation. Stage 2 focused on the final inference pipeline with calibration, test-time augmentation, final evaluation, and mobile application integration.
