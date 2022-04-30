#include <stdio.h>
#include <iostream>
using namespace std;
int check(int *pnt)
{
    // *pnt += 1;
    int *pn = new int;
    *pn = 2;
    pnt = pn;
    cout << &pnt << ' ' << pnt << ' ' << *pnt << endl;
    return 1;
}
int main()
{
    int *pointer = new int;
    *pointer = 1;
    cout << &pointer << ' ' << pointer << ' ' << *pointer << endl;
    check(pointer);
    cout << &pointer << ' ' << pointer << ' ' << *pointer << endl;
}