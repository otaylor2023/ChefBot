| model | sightings | named right | wrong name | not named | false finds per picture | s per picture |
|---|---|---|---|---|---|---|
| vlm (saved answers; its frames only) | 127 | 78% | 17% | 5% | 0.6 | 0.00 |
| owl (as the explorer uses it) | 758 | 43% | 36% | 21% | 1.6 | 1.62 |
| owl boxes + siglip names | 758 | 50% | 23% | 27% | 1.3 | 2.07 |
| siglip naming the true boxes (naming only) | 758 | 56% | 29% | 15% | 0.0 | 3.71 |

Named right, by visible pixels:

| model | 0-100 | 100-300 | 300-1000 | 1000-3000 | 3000-more |
|---|---|---|---|---|---|
| vlm (saved answers; its frames only) | 17% (6) | 73% (22) | 75% (61) | 91% (23) | 100% (15) |
| owl (as the explorer uses it) | 13% (171) | 39% (213) | 45% (234) | 78% (99) | 93% (41) |
| owl boxes + siglip names | 29% (171) | 52% (213) | 63% (234) | 47% (99) | 56% (41) |
| siglip naming the true boxes (naming only) | 39% (171) | 61% (213) | 68% (234) | 51% (99) | 49% (41) |

Most frequent wrong names:

- vlm (saved answers; its frames only): cup as mug (5), mug as cup (4), apple as tomato (2), pan as bowl (2), bottled water as bottle (1), bowl as plate (1)
- owl (as the explorer uses it): cup as mug (28), bottled water as condiment bottle (28), lemon as pear (23), mug as cup (21), banana as sponge (20), lemon as avocado (17)
- owl boxes + siglip names: pan as saucepan (41), lemon as lime (32), mug as cup (11), bottled water as bottled drink (7), cup as coffee cup (6), lemon as lettuce (5)
- siglip naming the true boxes (naming only): pan as saucepan (42), lemon as lime (34), mug as cup (9), cup as coffee cup (8), bottled water as bottled drink (7), mug as burrito (6)
