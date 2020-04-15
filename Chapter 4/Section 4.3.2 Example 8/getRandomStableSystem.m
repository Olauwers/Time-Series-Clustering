function sys = getRandomStableSystem(np,nz,nk)
    % Function to generate stable discrete SISO transferfunction models.
    % Parameters:
    % np: The number of poles
    % nz: The number of zeros
    % nk: The delay
    % COPYRIGHT 2019 Bob Vergauwen.
    
    
    % Generate poles:
    D_min = .1;
    D_max = .99;

    ph_min = pi/6;
    ph_max = pi-ph_min;
    
    % Check if  we need to add  a exta  real pole
    if (mod(np,2)==1)
        p = D_min + rand()*(D_max-D_min);
        np = np-1;
    else
        p=[];
    end

    damping = D_min + rand(np/2,1)*(D_max-D_min);
    ph = ph_min + rand(np/2,1)*(ph_max-ph_min);

    p = [p;damping.*exp(ph*1j);damping.*exp(-ph*1j)];
    p = [p;zeros(nk,1)];

    % Generate zeros:
    D_min = .1;
    D_max = .99;

    ph_min = pi/6;
    ph_max = pi-ph_min;
    
    % Check if  we need to add  a exta  real zero
    if (mod(nz,2)==1)
        z = D_min + rand()*(D_max-D_min);
        nz = nz-1;
    else
        z=[];
    end

    damping = D_min + rand(nz/2,1)*(D_max-D_min);
    ph = ph_min + rand(nz/2,1)*(ph_max-ph_min);

    z = [z;damping.*exp(ph*1j);damping.*exp(-ph*1j)];

    sys = zpk(z,p,1,1);