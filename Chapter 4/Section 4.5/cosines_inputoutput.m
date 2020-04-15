function cosines = cosines_inputoutput(U1,Y1,U2,Y2)

Y1onU1perp = projected(U1,Y1);
U1onY1perp = projected(Y1,U1);
Y2onU2perp = projected(U2,Y2);
U2onY2perp = projected(Y2,U2);

cosines = cosines_lq([Y1onU1perp;U2onY2perp],[Y2onU2perp;U1onY1perp]);