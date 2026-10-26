| model | sightings | named right | wrong name | not named | false finds per picture | s per picture |
|---|---|---|---|---|---|---|
| vlm (saved answers; its frames only) | 127 | 78% | 17% | 5% | 0.6 | 0.00 |
| vlm (saved answers; its frames only), aliases | 127 | 86% | 9% | 5% | 0.6 | 0.00 |
| owl (as the explorer uses it) | 758 | 43% | 36% | 21% | 1.6 | 0.99 |
| owl (as the explorer uses it), aliases | 758 | 50% | 30% | 21% | 1.6 | 0.00 |
| hybrid: owl, siglip under 1000 px | 758 | 54% | 24% | 23% | 1.5 | 0.98 |
| hybrid: owl, siglip under 1000 px, aliases | 758 | 63% | 15% | 23% | 1.5 | 0.00 |

Named right, by visible pixels:

| model | 0-100 | 100-300 | 300-1000 | 1000-3000 | 3000-more |
|---|---|---|---|---|---|
| vlm (saved answers; its frames only) | 17% (6) | 73% (22) | 75% (61) | 91% (23) | 100% (15) |
| vlm (saved answers; its frames only), aliases | 17% (6) | 82% (22) | 89% (61) | 91% (23) | 100% (15) |
| owl (as the explorer uses it) | 13% (171) | 39% (213) | 45% (234) | 78% (99) | 93% (41) |
| owl (as the explorer uses it), aliases | 16% (171) | 43% (213) | 60% (234) | 79% (99) | 93% (41) |
| hybrid: owl, siglip under 1000 px | 29% (171) | 52% (213) | 56% (234) | 78% (99) | 93% (41) |
| hybrid: owl, siglip under 1000 px, aliases | 32% (171) | 62% (213) | 74% (234) | 79% (99) | 93% (41) |

Most frequent wrong names:

- vlm (saved answers; its frames only): cup as mug (5), mug as cup (4), apple as tomato (2), pan as bowl (2), milk as bottled drink (1), lemon as avocado (1)
- vlm (saved answers; its frames only), aliases: apple as tomato (2), pan as bowl (2), milk as bottled drink (1), lemon as avocado (1), bowl as plate (1), spoon as knife (1)
- owl (as the explorer uses it): cup as mug (28), bottled water as condiment bottle (28), lemon as pear (23), mug as cup (21), banana as sponge (20), lemon as avocado (17)
- owl (as the explorer uses it), aliases: bottled water as condiment bottle (28), lemon as pear (23), banana as sponge (20), lemon as avocado (17), spoon as knife (15), lemon as sponge (14)
- hybrid: owl, siglip under 1000 px: lemon as lime (32), mug as cup (12), cup as mug (11), bottled water as condiment bottle (11), pan as strainer (6), pan as saucepan (6)
- hybrid: owl, siglip under 1000 px, aliases: bottled water as condiment bottle (11), pan as strainer (6), spoon as banana (5), apple as tomato (5), apple as cherry (5), lemon as lettuce (5)
