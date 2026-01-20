%SPAI.M --- Choice of Grote+Huckle 97 (method=1) // Gould+Scott 98 (method=2)
%Usage:   M = spai(A,nzmax,tol,method,B,pattern)
%   or    M = spai(A,nzmax,tol,method)
%   or    M = spai(A,nzmax)          % b_j=e_j is the usual SPAI (default)
%   or    M = spai(A)      % for SPAI approximation of  A*M=B or A m_j = b_j
%where  1/2/3: A/nzmax/tol : Input matrix A for M with nnzmax per col and TOL
%   Option  4: method=1 GH appoach (default) or 2 for GS appoach
%           5: B matrix for the option of A*M=B (if B is not default eye(n))
%           6: pattern = pattern matrix (e.g. eye(n)) which may be []
% e.g.  A=cz(16,7)+rand(16);  M=spai2(A,5,0.1);
%  or   A=sprand(24,24,.4)+speye(24); M=spai2(A,5,0.1,2);
%
% Martyn D. Hughes and Ke Chen (University of Liverpool, UK)    (spai3.m based)

function M = spai(A,nz_max,tol,method,B,patt,Adpa)
if nargin<1, help spai, return, end  %Adap=0 for no adaptive searching
%+++++++++++++++++Step 1.  Set up and inputs +++++++++++++++++++++++++++++
    count=1; iout=0; it_use=0;
    n=size(A,1); col_sum2=sum( A.^2 );
if nargin<2, nz_max=50; nnzmax=50; else, nnzmax=nz_max; end
if nargin<3, tol=0.4; end
if nargin<4, method=1; end
if nargin<5, B=speye(n); end  % Am_j=e_j /AM=I or AM=B
if nargin<6, patt=speye(n); end
if nargin<7, Adpa=1;        end
verbose = 0;
    if length(B)<1,B=speye(n);end, if length(patt)<1,patt=speye(n);end

%   if nnzmax>n,nnzmax=round(n/2);end
    s=min(round(nnzmax/2),5);  % Number of new cols per step
    if s>n, s=round(n/4); end

    IT_max = max(ceil(nnzmax/s),3); % In theory, infy as tol is active
    %IT_max = 20;
    if Adpa==0; IT_max=1; end %%%%%%% Switch off adaptive SPAI (rely on patt)

       cpu0=cputime;
  count=1; color_list='rcmbgy'; s_list='^>v<ox';
   iout=1; if n>24, iout=0; end %% May overide to show graph ...

ehat=[]; %set ehat - might not need this

     if iout==1
       figure; spy(A), hold on
     end

if verbose
    disp(['Starting ' mfilename '.m for tol=' num2str(tol) ' s=' num2str(s)...
     ' nnzmax =' num2str(nnzmax) ' N=' num2str(n) ' Method ' num2str(method)...
     ' A*M=B for M'])
        M=sparse(n,n); %set M
end

%+++++++++++++++Step 2 Start columns. Outer loop+++++++++++++++++++++++++++++

for k=1:n%start of columns % Outer Loop
%cpu1=cputime;flop1=flops;

    c_now=1;  %figure

%reset the vectors
    curlyj=sparse(n,1); curlyi=sparse(n,1); 
    curlyjhat=[]; curlyihat=[]; % new 

    mk=sparse(n,1); ehat=B(:,k); % ehat = full length ek (or B_k in general)

%+++++++++++++++Step 3 Find indices of nonzero entries in kth col of m, mk, 
%                          and put in curlyj++++++++++++++++++++++++++++++++++  
            
  curlyj=find(patt(:,k)); %curlyj = col vec with indices of nz entries of mk
   
    if isempty(curlyj)==1
        curlyj=[k];
    end
    rk=sparse(n,1); normrk=1.2;
    
%++++++++Step 5 Indices of nonzero entries in curlyj cols of A are put in curlyi++++++++++++++
        [ii,ji]=find(A(:,curlyj)); curlyi=unique(ii); % Row index of J cols
        if isempty(curlyi)==1
        curlyi=[k];
        end

%+++++++++++++++Step 4 While residual norm > tolerance. Inner loop++++++++++++
    iteration=0; Q=[]; space='                       '; nx=[]; ny=[];
    maxi=ceil(log(max(nnzmax,10000))/log(10));

%% Adaptive SPAI / QR NOTATION
 %  chat --- active RHS (dim increasing)
 %  ohat --- residual half of RHS (appending 0's)
 %  R    --- R for QR increases
 %  Q_j  --- Q_j for Q in step j=iteration of QR (dim's different & increasing)
 %  nx   --- collection of 1st dim n1
 %  ny   --- collection of 2nd dim n2

while iteration<IT_max & nnz(mk)<nnzmax & normrk>tol %% LOOP tol for col "k"
        iteration=iteration+1;

        rhojreduced=[]; %NEED TO RESET ALL THE VECTORS BAR CURLYJ
             curlyl=sparse(n,1); curlyn=sparse(n,1);
        
if isempty(curlyjhat)~=1 & iout==1
 r_long=sparse(size(A));  r_long(curlyjhat,k)=curlyjhat;
                          r_long(curlyj,k)=curlyj;
 spy(r_long,[s_list(c_now) color_list(c_now)]);  hold on
 c_now=c_now+1; if c_now>6, c_now=1; end 
end
      
                %%------------------------------------------------------
if iteration==1 %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  
                %%------------------------------------------------------
%++++++++Step 6 Set ehat, ahat corr. to curlyi rows and curlyj cols of A+++++++
%++++++++(nonzero entries of A corresponding to the nonzero indices in mk)+++++
        ehat= full( ehat(curlyi) );
        ahat= full( A(curlyi,curlyj) );
%++++++++Step 7 Solve the LS problem to minimize the error in the AI ++++++++++ 
                     % Use the Householder method for QR reduction 
[n1 n2]=size(ahat);  % this is vector with m and n for ahat 
[Q1 R] = qr(ahat); ehat=Q1'*ehat;

% now need to get the upper square part of R as before
    chat = ehat(1:n2);       % This is RHS
    ohat = ehat((n2+1):n1);  % This is RHS extra(r) in total R(1:n1,n2+1)
       R = R(1:n2,1:n2);
       mkhat=R\chat;         % this should be the solution

    id = floor(log(iteration)/log(10));  % Save Qj=Q1 
    id=sprintf('Q%d%s',iteration,space(1:maxi-id)); 
                                                     Q=[Q;id];
    nx(iteration) = n1; ny(iteration) = n2;

     %% ----------------------------------------------------------------
else %% iteration %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR 
     %% ----------------------------------------------------------------
                         %% A_new = [ A^  A(I, J^)   = [ A^  ahat
                         %%           0   A(I^,J^) ]     0   ahat ]
         curlyi=[curlyi(:); curlyihat];%added to curlyi
         curlyj=[curlyj(:); curlyjhat];%added to curlyj  (don't do unique)
      n1h = length(curlyihat); % New col induced rows' indices
      n2h = length(curlyjhat); % New cols
      nx(iteration) = nx(iteration-1) + n1h;
      ny(iteration) = ny(iteration-1) + n2h; % Mr ehat below is the full RHS
                                            % which has gone thru past Q_j's
        ehat=[ehat(:); zeros(n1h,1)];      % Extend rhs to the new length
        ahat= full( A(curlyi,curlyjhat) ); % as I^ is non-zero.  Box up+RHS
        
        for j=1:iteration-1   % Apply prev Q_j's onto ahat and rhs
           n1=1; if j>1, n1=ny(j-1)+1; end
           range = n1:nx(j) ; 
           id=sprintf('ahat(range,:)= transpose(%s) * ahat(range,:);',Q(j,:));
           eval(id)
        end
            n1 = ny(iteration-1)+1;    % Start of current QR portion
              B1 = ahat(1:n1-1,1:n2h); % Grote/Huckle top (B1)
           range = n1:nx(iteration);   % Current QR (n2h cols)
            ahat = ahat(range,1:n2h);  % Grote/Huckle bot (B2)

        [QH RH]=qr( ahat ); RH=RH(1:n2h,1:n2h);
                   ehat(range)=QH'*ehat(range);
           n2 = floor(log(iteration)/log(10));  % Save Qj=QH
            n2 = floor(log(iteration)/log(10));  % Save Qj=QH
            id=sprintf('Q%d%s',iteration,space(1:maxi-n2)); Q=[Q;id];
            id=sprintf('Q%d = QH;',iteration); eval(id) 

          n2 = ny(iteration-1);
          R = [R              B1
               sparse(n2h,n2) RH];  % QR update - Upper R updated
          n2 = ny(iteration);
        chat = ehat(1:n2);     % This is RHS

       mkhat=R\chat;          % This should be the solution 

     %% ----------------------------------------------------------------
end  %iteration>1  % QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR  %% QR 
     %% ----------------------------------------------------------------

%++++++++++Step 13 Retain s best ones, add to curlyj, do inner loop again++++
      
         curlyjhat=[]; curlyihat=[]; % new 

%+++++++++Step 8 Update the kth col of whole matrix M++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%+++++++++Step 9 Start augmentation process to add additional entries to mk++++
    
        ar=A(:,curlyj); % reduced matrix A(.,curlyj) 
        rk=ar*mkhat;    % giving the residual of the least squares problem
        rk=B(:,k)-rk;   % Allow Am_k=b_k as well as Am_k=e_k !!!!!!!!!!!!!!
        normrk=norm(rk);%                  with the current reduced matrix
   curlyl=find(rk);  [in,jn]=find(A(curlyl,:)); curlyn=unique(jn);   
curlyjhat=setdiff(curlyn,curlyj);

            mk=sparse(n,1); mk(curlyj)=mkhat;  M(:,k)=mk;
       if isempty(curlyjhat)==1   % No improvement is possible ......
            break
       end %end if
        
%++++++++++Step 10  Check the norm of the new residuals for each candidate ++++
%++++++++++Step 11 Calculate the mean of the residual norms+++++++++++++++++++

if length(curlyjhat) > s % otherwise no need to scan
      rhoj=[]; %reset rhoj
for j=1:length(curlyjhat) %% SCAN
       jj=curlyjhat(j);  ccc=A(:,jj);         %A*ej, the jth col of A
   if method == 1 %% GH  GH  GH  GH  GH  GH  GH  GH  GH  Grote-Huckle'97
      rhoj(j)=(rk'*ccc)^2/col_sum2(jj); % look for the max

   else %---------%% GS  GS  GS  GS  GS  GS  GS  GS  GS  Gould-Scott'98
        % Repeat previous QR update for a candidate vector  ccc for rho!!!!
        jj=curlyjhat(j);  ccc=A(:,jj);         %A*ej, the jth col of A
     jhat = find(ccc); % In case there are new ones beyond curlyi
     ihat = setdiff(jhat,curlyi); % extra rows in c via jj --> amend r for c^Tr
     n1h = length(ihat); % mimick before
     n2h = 1;            % Surely jj is not on previous curlyj list
     nxc = nx; nxc(iteration+1)=nxc(iteration)+n1h;
     nyc = ny; nyc(iteration+1)=nyc(iteration)+n2h;
        ahat= full(A([curlyi(:); ihat(:)],jj) ); % New long compact vector
        
        for L=1:iteration %Apply prev Q_L's onto ahat (iteration steps)
           n1=1; if L>1, n1=nyc(L-1)+1; end
           range = n1:nxc(L) ; 
           id=sprintf('ahat(range,:)= transpose(%s) * ahat(range,:);',Q(L,:));
           eval(id)
        end
            n1 = nyc(iteration)+1;     % Start of current QR portion
              B1 = ahat(1:n1-1,1:n2h); % Grote/Huckle top (B1)
           range = n1:nxc(iteration+1);% Current QR (n2h cols)
            ahat = ahat(range,1:n2h);  % Grote/Huckle bot (B2)
             rho = norm(ahat);         % [QH RH]=qr( ahat ); rho=RH(n2h,n2h); 
     rhoj(j)=(rk'*ccc)^2 / rho^2;      % Shall look for the max
   end %if---------%% --------------------------------------------------
end %end j
      [rhoj k_index]=sort(-rhoj);  % Negative for max at front
      curlyjhat=curlyjhat(k_index(1:s)); %The most profitable one just found
end % if > s

% -- -------------------------------------------------------------------
     n2h=length(curlyjhat);
   if n2h~=0
     hata = A(:,curlyjhat);
     curlyihat =[];
     for col=1:n2h
       curlyihat = [ curlyihat; find(hata(:,col)) ];
     end
       curlyihat = unique( setdiff( curlyihat, curlyi ) );
   end
      
end %END while loop for tol (Inner loop) and each col "k"

%if k>n % off
%        id=['spai  COL k ' num2str(k) ' iter=' num2str(iteration) ' flop='...
%            num2str(flops-flop1) ' Cpu=', num2str(cputime-cpu1) ]; disp(id)
%end 

if verbose
    if rem(k-1,round(n/5))==0
    fprintf('\t %3d%% Done <%s.m> \t (Adaptive=%d)\n',round(k/n*100),mfilename,Adpa)
    end
end

end %%%%%% (Outer loop) + <> + <> + <> + <> + <> + <> + <> + <> + <> + <> + <> +
        Err=norm(A*M-B,'fro')/sqrt(n);
        End=' Yes '; if Err>tol, End=' No! '; end
%         id=sprintf('%3.2g ',flops);
%         id=[mfilename ': TOL ' num2str(tol) ' s=' num2str(s) ' flop='...
%             id 'Cpu=', num2str(cputime-cpu0) ' M_{nnz}='...
%             num2str(nnz(M)) '  OK?' End];
     if iout==1
        ylabel('Efficient QR version spai2.m')
        xlabel(['err_F = ' num2str( Err )])
        t=title(id);set(t,'color','r')
     end 
     if verbose
        disp([id ' err_F = ' num2str( Err )])
     end