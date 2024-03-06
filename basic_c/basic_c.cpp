// basic_c.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctype.h>

/* count digits, white space, others */
void kr_array_example() {
    std::cout << "Running kr_array_example() ..." << std::endl;

    int c, i, nwhite, nother;
    int ndigit[10];
    nwhite = nother = 0;
    for (i = 0; i < 10; ++i)
        ndigit[i] = 0;
    while ((c = getchar()) != EOF)
        if (c >= '0' && c <= '9')
            ++ndigit[c - '0'];
        else if (c == ' ' || c == '\n' || c == '\t')
            ++nwhite;
        else
            ++nother;
    std::cout << "digits = ";
    for (i = 0; i < 10; ++i)
        std::cout << ndigit[i] << " ";
    std::cout << ", white space = " << nwhite << ", other = " << nother << std::endl;
}

/* Pointers & addresses */
void kr_5_1_pointers_and_addresses() {
    std::cout << "kr 5.1 pointers and addresses" << std::endl;
    char c = '1';
    char* p = &c; // p points to c
    std::cout << "c = " << c << std::endl;
    std::printf("p = %X\n", p);
    std::cout << "*p = " << *p << std::endl;
    
    std::cout << std::flush;

    /* */
    int x = 1, y = 2, z[10];
    int* ip; /* ip is a pointer to int */
    ip = &x; /* ip now points to x */
    y = *ip; /* y is now 1 */
    *ip = 0; /* x is now 0 */
    ip = &z[0]; /* ip now points to z[0] */

    /* */
    double* dp, atof(char*);
    z[0] = 3;
    *ip = *ip + 10; // or *ip += 10
    y = *ip + 1;
    *ip += 1;
    ++*ip;
    (*ip)++;
    int *iq = ip;
    double d = 1.0f;
    dp = &d;

    return;
}

void swap_1(int x, int y) /* WRONG */
{
    int temp;
    temp = x;
    x = y;
    y = temp;
}

void swap_2(int* px, int* py) /* interchange *px and *py */
{
    int temp;
    temp = *px;
    *px = *py;
    *py = temp;
}

#define BUFSIZE 100
char buf[BUFSIZE];
int bufp = 0;

int getch(void) /* get a (possibly pushed-back) character */
{
    return (bufp > 0) ? buf[--bufp] : getchar();
}

void ungetch(int c) /* push character back on input */
{
    if (bufp >= BUFSIZE)
        printf("ungetch: too many characters\n");
    else
        buf[bufp++] = c;
}

/* getint: get next integer from input into *pn */
int getint(int* pn)
{
    int c, sign;
    while (isspace(c = getch())) /* skip white space */
        ;
    if (!isdigit(c) && c != EOF && c != '+' && c != '-') {
        ungetch(c); /* it is not a number */
        return 0;
    }
    sign = (c == '-') ? -1 : 1;
    if (c == '+' || c == '-')
        c = getch();
    for (*pn = 0; isdigit(c); c = getch())
        *pn = 10 * *pn + (c - '0');
    *pn *= sign;
    if (c != EOF)
        ungetch(c);
    return c;
}

#define SIZE 20

void kr_5_2_pointers_and_function_arguments() {
    std::cout << "kr 5.2 pointers and function arguments" << std::endl;
    int n, array[SIZE], getint(int*);
    for (n = 0; n < SIZE && getint(&array[n]) != EOF; n++)
        ;
}

/* strlen: return length of string s */
int strlen(char* s)
{
    int n;
    for (n = 0; *s != '\0'; s++)
        n++;
    return n;
}

void f(int arr[]) {
    std::cout << arr[0] << std::endl;
}

void kr_5_3_pointers_arrays() {
    std::cout << "kr 5.3 pointers and arrays" << std::endl;

    char array[100];
    char* ptr = array;
    int a[10];
    int* pa;
    pa = &a[0];

    int x = *pa;
    *(pa + 1);

    for (int i = 0; i < 100; i++) {
        array[i] = i + 1;
    }

    for (int i = 0; i < 10; i++) {
        a[i] = i + 1;
    }

    strlen("hello, world"); /* string constant */
    strlen(array); /* char array[100]; */
    strlen(ptr); /* char *ptr; */

    char* s;
    f(&a[2]);
    f(a + 2);
}

#define ALLOCSIZE 10000 /* size of available space */
static char allocbuf[ALLOCSIZE]; /* storage for alloc */
static char* allocp = allocbuf; /* next free position */

char* alloc(int n) /* return pointer to n characters */
{
    if (allocbuf + ALLOCSIZE - allocp >= n) { /* it fits */
        allocp += n;
        return allocp - n; /* old p */
    }
    else /* not enough room */
        return 0;
}

void afree(char* p) /* free storage pointed to by p */
{
    if (p >= allocbuf && p < allocbuf + ALLOCSIZE)
        allocp = p;
}

void kr_5_4_address_arithmetic() {
    std::cout << "kr 5.4 address arithmetic" << std::endl;
}

#define COL_SIZE 32
#define ROW_SIZE 4

static void row_major(char a[][COL_SIZE]) {
    int count = 0;
    for (int r = 0; r < ROW_SIZE; r++) {
        for (int c = 0; c < COL_SIZE; c++) {
            a[r][c] = count++;
        }
    }
    a;
}

void col_major(char a[][COL_SIZE]) {
    int count = 0;
    for (int c = 0; c < COL_SIZE; c++) {
        for (int r = 0; r < ROW_SIZE; r++) {
            a[r][c] = count++;
        }
    }
    a;

}

int main()
{
    // kr_array_example();
    kr_5_1_pointers_and_addresses();
    kr_5_2_pointers_and_function_arguments();
    kr_5_3_pointers_arrays();
    char a[ROW_SIZE][COL_SIZE];
    row_major(a);
    col_major(a);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
