# weekly-pattern-in-stock-markets
exploiting weekly market patterns for stock trading

In this repository we will compile a couple of jupyter-notebooks, generated in Google-Colab, containing code for strategies that try to identify weekly pattern in stock markets for usage in stock trading. 

# 1. 16-weeks-cycle strategy for German Stock market
The strategy was generated and first published by Thomas Gebert, a german stock trader and publicist (see: Thomas Gebert, Kurzfristige Strategien fuer Anleger, Boersenbuchverlag, 2020 (in German)).  He investigated into a potential 16-weeks-cycle, observeable in the closing-prices of the DAX-Index. Within this 16-weeks-cycle a fixed pattern of trading positions per week is applied to go LONG, SHORT or to stay FLAT. The notebook repeats a slightly simplified version of his studies (analyzing the years 2000 to 2019, that were covered by the publication). Next, the analyses was extended to end of february 2024. In a stratgy modification a small change in the entry/exit timing is proposed. Finally a sensitivity analyses is provided by shifting the identified pattern across the 16-weeks-cycle. 
The results of the publication could be confirmed, however it turns out, that this strategy degrades significantly after 2019. Therefore the published strategy should not be used in the current market conditions. 

# 2. Pattern detection and optimisation within N-weeks-cycles using Genetic Algorithms
## 2.1. 16-weeks-cycle 
A pattern detection algorithm was tested and implemented. For the 16 weeks cycle a further optimsiation was done by optimizing the weekly-pattern of trade positions by Genetic Algorithms

## 2.2 N-weeks-cycle
The analyses of the 16 weeks cycle is extended to general N-weeks-cycles. 

## 2.3. adding fee
A scheme of fees per trade (fixed fee and variable fee) was applied. This will limit use of vectorisation in python. The equity calculation was coded using just in time compilation (from numba-module) for the loops in the core of the code. This helped to minimize the additional calculation time. It turned out, that the consideration of fees significantlz reduced the performance of the system

# 3. Walking Forward
During the analyses a more flexible adaptive aproach was tested. The parameters were optimized over a period of M-years and the applied to the current year. The M-years window is shifted year by year over the total-time-periode. Therefore each year a new set of (optimized) parameters are calculatd and aplied. This will enable a certain ammount of adaptability. 
   
