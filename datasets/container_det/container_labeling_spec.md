# Container Labeling Spec

This spec defines the first supervised dataset for supermarket container detection. It is optimized for the baseline in `shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml`.

## Goal

Annotate visible containers that can receive, hide, or legitimately hold merchandise in CCTV frames. The detector only predicts appearance classes. Risk meaning is handled later by `object_container`, tracking, store ROI, and checkout context.

## Classes

| id | class | role | Label when |
|---:|---|---|---|
| 0 | `bag` | private | Generic personal or unknown soft container, tote, paper bag, plastic shopping bag before checkout context is known, open bag, purse-like bag that is not clearly handbag/backpack. |
| 1 | `backpack` | private | Backpack, sling backpack, school bag, or shoulder-carried backpack shape. |
| 2 | `handbag` | private | Handbag, purse, clutch, small tote carried by hand or arm. |
| 3 | `suitcase` | private | Suitcase, luggage, rolling luggage, travel case. |
| 4 | `basket` | normal shopping | Store shopping basket, handheld retail basket, wheeled shopping basket if not large enough to be a cart. |
| 5 | `cart` | normal shopping | Store shopping cart or trolley, including partially visible cart baskets and child-seat carts. |
| 6 | `plastic_bag` | normal shopping | Plastic bags used for bulk sales in supermarkets. |
| 7 | `stroller` | special | Baby stroller or pram. Label the merchandise-holding body region, not the whole person. |
| 8 | `helmet` | special | Motorcycle/bicycle helmet or hard-shell helmet that can receive merchandise. Label the shell/opening body, not the wearer's head if worn normally. |
