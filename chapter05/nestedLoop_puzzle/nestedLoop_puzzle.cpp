// nestedLoop_puzzle.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <time.h>
#include "common.h"

#define ROWS 1000
#define COLUMNS 1000
int m[ROWS][COLUMNS];


int main()
{
    std::cout << "nested loop puzzle...\n";
    int i, j;
    double d = 0;
    std::chrono::steady_clock::time_point begin;

    int count = 1;
    begin = StartTimer();
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLUMNS; j++) {
            m[i][j] = count++;
        }
    }
    std::cout << "Run time of row major order is " 

              << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" 
              << std::endl;

    count = 1;
    begin = StartTimer();
    for (j = 0; j < COLUMNS; j++) {
        for (i = 0; i < ROWS; i++) {
            m[i][j] = count++;
        }
    }
    
    std::cout << "Run time of column major order is " 
              << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" 
              << std::endl;    
    exit(0);
}