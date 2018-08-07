function cints(ctx,res) {  
	/* 
	Return MLEs for random event process [ {x,y,...}, ...] given ctx parameters
	*/
		
		function coherenceIntervals(solve, cb) { // unsupervised learning of coherence intervals M, SNR, etc
		/*
			H[k] = observation freqs at count level k
			T = observation time
			N = number of observations
			solve = {use: "lma" | ...,  lma: [initial M], lfa: [initial M], bfs: [start, end, increment M] }
			callback cb(unsupervised estimates)
		*/
			function logNB(k,a,x) { // negative binomial objective function
			/*
			return log{ p0 } where
				p0(x) = negbin(a,k,x) = (gamma(k+x)/gamma(x))*(1+a/x)**(-x)*(1+x/a)**(-k) 
				a = <k> = average count
				x = script M = coherence intervals
				k = count level
			 */
				var
					ax1 =  1 + a/x,
					xa1 = 1 + x/a,

					// nonindexed log Gamma works with optimizers, but slower than indexed versions
					logGx = GAMMA.log(x),
					logGkx = GAMMA.log(k+x), 
					logGk1 = GAMMA.log(k+1);

					// indexed log Gamma produce round-off errors in optimizers 
					// logGx = logGamma[ floor(x) ],
					// logGkx = logGamma[ floor(k + x) ],
					// logGk1 = logGamma[ floor(k + 1) ];

				return logGkx - logGk1 - logGx  - k*log(xa1) - x*log(ax1);
			}

			function LFA(init, f, logp) {  // linear-factor-analysis (via newton raphson) for chi^2 extrema - use at your own risk
			/*
			1-parameter (x) linear-factor analysis
			k = possibly compressed list of count bins
			init = initial parameter values [a0, x0, ...] of length N
			logf  = possibly compressed list of log count frequencies
			a = Kbar = average count
			x = M = coherence intervals		
			*/

				function p1(k,a,x) { 
				/*
				return p0'(x) =
							(1 + x/a)**(-k)*(a/x + 1)**(-x)*(a/(x*(a/x + 1)) - log(a/x + 1)) * gamma[k + x]/gamma[x] 
								- (1 + x/a)**(-k)*(a/x + 1)**(-x)*gamma[k + x]*polygamma(0, x)/gamma[x] 
								+ (1 + x/a)**(-k)*(a/x + 1)**(-x)*gamma[k + x]*polygamma(0, k + x)/gamma[x] 
								- k*(1 + x/a)**(-k)*(a/x + 1)**(-x)*gamma[k + x]/( a*(1 + x/a)*gamma[x] )			

						=	(1 + x/a)**(-k)*(a/x + 1)**(-x)*(a/(x*(a/x + 1)) - log(a/x + 1)) * G[k + x]/G[x] 
								- (1 + x/a)**(-k)*(a/x + 1)**(-x)*PSI(x)*G[k + x]/G[x] 
								+ (1 + x/a)**(-k)*(a/x + 1)**(-x)*PSI(k + x)*G[k + x]/G[x] 
								- k*(1 + x/a)**(-k)*(a/x + 1)**(-x)*G[k + x]/G[x]/( a*(1 + x/a) )			

						=	G[k + x]/G[x] * (1 + a/x)**(-x) * (1 + x/a)**(-k) * {
								(a/(x*(a/x + 1)) - log(a/x + 1)) - PSI(x) + PSI(k + x) - k / ( a*(1 + x/a) ) }

						= p(x) * { (a/x) / (1+a/x) - (k/a) / (1+x/a) - log(1+a/x) + Psi(k+x) - Psi(x)  }

						= p(x) * { (a/x - k/a) / (1+x/a) - log(1+a/x) + Psi(k+x) - Psi(x)  }

					where
						Psi(x) = polyGamma(0,x)
				 */
					var
						ax1 =  1 + a/x,
						xa1 = 1 + x/a,

						// indexed Psi may cause round-off problems in optimizer
						psix = Psi[ floor(x) ], 
						psikx = Psi[ floor(k + x) ], 

						slope = (a/x - k/a)/ax1 - log(ax1) + psikx - psix;

					return exp( logp(k,a,x) ) * slope;  // the slope may go negative so cant return logp1		
				}

				function p2(k,a,x) {  // not used
				/*
				return p0" = 
						(1 + x/a)**(-k)*(a/x + 1)**(-x)*( a**2/(x**3*(a/x + 1)**2) 
							+ (a/(x*(a/x + 1)) - log(a/x + 1))**2 - 2*(a/(x*(a/x + 1)) - log(a/x + 1) )*polygamma(0, x) 
						+ 2*(a/(x*(a/x + 1)) - log(a/x + 1))*polygamma(0, k + x) 
						+ polygamma(0, x)**2 
						- 2*polygamma(0, x)*polygamma(0, k + x) + polygamma(0, k + x)**2 - polygamma(1, x) + polygamma(1, k + x) 
						- 2*k*(a/(x*(a/x + 1)) - log(a/x + 1))/(a*(1 + x/a)) + 2*k*polygamma(0, x)/(a*(1 + x/a)) 
						- 2*k*polygamma(0, k + x)/(a*(1 + x/a)) + k**2/(a**2*(1 + x/a)**2) + k/(a**2*(1 + x/a)**2))*gamma(k + x)/gamma(x);
				 */
					var
						ax1 =  1 + a/x,
						xa1 = 1 + x/a,
						xak = xa1**(-k),
						axx = ax1**(-x),

						// should make these unindexed log versions
						gx = logGamma[ floor(x) ],
						gkx = logGamma[ floor(k + x) ],

						logax1 = log(ax1),
						xax1 = x*ax1,
						axa1 = a*xa1,				

						// should make these Psi 
						pg0x = polygamma(0, x),
						pg0kx = polygamma(0, k + x);

					return xak*axx*(a**2/(x**3*ax1**2) + (a/xax1 - logax1)**2 - 2*(a/xax1 - logax1)*pg0x 
								+ 2*(a/xax1 - logax1)*pg0kx + pg0x**2 
								- 2*pg0x*pg0kx + pg0kx**2 - polygamma(1, x) + polygamma(1, k + x) 
								- 2*k*(a/xax1 - logax1)/axa1 + 2*k*pgx/axa1 - 2*k*pg0kx/axa1 
								+ k**2/(a**2*xa1**2) + k/(a**2*xa1**2))*gkx/gx;
				}

				function chiSq1(f,a,x) { 
				/*
				return chiSq' (x)
				*/
					var 
						sum = 0,
						Kmax = f.length;

					for (var k=1; k<Kmax; k++) sum += ( exp( logp0(a,k,x) ) - f[k] ) * p1(a,k,x);

					//Log("chiSq1",a,x,Kmax,sum);
					return sum;
				}

				function chiSq2(f,a,x) {
				/*
				return chiSq"(x)
				*/
					var
						sum =0,
						Kmax = f.length;

					for (var k=1; k<Kmax; k++) sum += p1(a,k,x) ** 2;

					//Log("chiSq2",a,x,Kmax,sum);
					return 2*sum;
				}

				var
					Mmax = 400,
					Kmax = f.length + Mmax,
					eps = $(Kmax, (k,A) => A[k] = 1e-3),
					Zeta = $(Kmax, (k,Z) => 
						Z[k] = k ? ZETA(k+1) : -0.57721566490153286060   // -Z[0] is euler-masheroni constant
					), 
					Psi1 = Zeta.sum(),
					Psi = $(Kmax, (x,P) =>   // recurrence to build the diGamma Psi
							P[x] = x ? P[x-1] + 1/x : Psi1 
					);

				return NRAP( (x) => chiSq1(f, Kbar, x), (x) => chiSq2(f, Kbar, x), init[0]);  // 1-parameter newton-raphson
			}

			function LMA(init, k, logf, logp) {  // levenberg-marquart algorithm for chi^2 extrema
			/*
			N-parameter (a,x,...) levenberg-marquadt algorithm where
			k = possibly compressed list of count bins
			init = initial parameter values [a0, x0, ...] of length N
			logf  = possibly compressed list of log count frequencies
			a = Kbar = average count
			x = M = coherence intervals
			*/

				switch ( init.length ) {
					case 1:
						return LM({  // 1-parm (x) levenberg-marquadt
							x: k,  
							y: logf
						}, function ([x]) {
							//Log(Kbar, x);
							return (k) => logp(k, Kbar, x);
						}, {
							damping: 0.1, //1.5,
							initialValues: init,
							//gradientDifference: 0.1,
							maxIterations: 1e3,  // >= 1e3 with compression
							errorTolerance: 10e-3  // <= 10e-3 with compression
						});
						break;

					case 2:

						switch ("2stage") {
							case "2parm":  // greedy 2-parm (a,x) approach will often fail when LM attempts an x<0
								return LM({  
									x: k,  
									y: logf  
								}, function ([x,u]) {
									Log("2stage LM",x,u);
									//return (k) => logp(k, Kbar, x, u);
									return x ? (k) => logp(k, Kbar, x, u) : (k) => -50;
								}, {
									damping: 0.1, //1.5,
									initialValues: init,
									//gradientDifference: 0.1,
									maxIterations: 1e2,
									errorTolerance: 10e-3
								});

							case "2stage":  // break 2-parm (a,x) into 2 stages
								var
									x0 = init[0],
									u0 = init[1],
									fit = LM({  // levenberg-marquadt
										x: k,  
										y: logf
									}, function ([u]) {
										//Log("u",u);
										return (k) => logp(k, Kbar, x0, u);
									}, {
										damping: 0.1, //1.5,
										initialValues: [u0],
										//gradientDifference: 0.1,
										maxIterations: 1e3,  // >= 1e3 with compression
										errorTolerance: 10e-3  // <= 10e-3 with compression
									}),
									u0 = fit.parameterValues[0],
									fit = LM({  // levenberg-marquadt
										x: k,  
										y: logf
									}, function ([x]) {
										//Log("x",x);
										return (k) => logp(k, Kbar, x, u0);
									}, {
										damping: 0.1, //1.5,
										initialValues: [x0],
										//gradientDifference: 0.1,
										maxIterations: 1e3,  // >= 1e3 with compression
										errorTolerance: 10e-3  // <= 10e-3 with compression
									}),
									x0 = fit.parameterValues[0];

								fit.parameterValues = [x0, u0];
								return fit;	
							}
						break;	
				}
			}

			function BFS(init, f, logp) {   // brute-force-search for chi^2 extrema
			/*
			1-parameter (x) brute force search
			k = possibly compressed list of count bins
			init = initial parameter values [a0, x0, ...] of length N
			logf  = possibly compressed list of log count frequencies
			a = Kbar = average count
			x = M = coherence intervals			
			*/
				function NegBin(NB, Kbar, M, logp) {
					NB.use( (k) => NB[k] = exp( logp(k, Kbar, M) ) );
				}

				function chiSquared(p, f, N) {
					var chiSq = 0, err = 0;
					p.use( (k) => {
						//chiSq += (H[k] - N*p[k])**2 / (N*p[k]);
						chiSq += (f[k] - p[k])**2 / p[k];
					});
					return chiSq * N;
				}

				var
					pRef = $(f.length),
					Mbrute = 1,
					chiSqMin = 1e99;

				for (var M=init[0], Mmax=init[1], Minc=init[2]; M<Mmax; M+=Minc) {  // brute force search
					NegBin(pRef, Kbar, M, logNB);
					var chiSq = chiSquared(pRef, fK, N);

					Log(M, chiSq, pRef.sum() );

					if (chiSq < chiSqMin) {
						Mbrute = M;
						chiSqMin = chiSq;
					}
				} 
				return Mbrute;
			}

			var
				/*
				logGamma = $(Ktop , function (k, logG) {
					logG[k] = (k<3) ? 0 : GAMMA.log(k);
				}),
				*/
				/*
				Gamma = $(Ktop, function (k,G) {
					G[k] = exp( logGamma[k] );
				}),
				*/
				H = solve.H,
				N = solve.N,
				T = solve.T,

				Nevs = 0, 	// number of events
				Kmax = H.length,  // max count
				Kbar = 0,  // mean count
				K = [],  // count list
				compress = solve.lfa ? false : true,   // enable pdf compression if not using lfa
				interpolate = !compress,
				fK = $(Kmax, function (k, p) {    // count frequencies
					if (interpolate)  {
						if ( H[k] ) 
							p[k] = H[k] / N;

						else
						if ( k ) {
							N += H[k-1];
							p[k] = H[k-1] / N;
						}

						else
							p[k] = 0;
					}
					else
						p[k] = H[k] / N;
				});

			//H.forEach( (h,n) => Log([n,h]) );

			H.use( (k) => {
				Kbar += k * fK[k];
				Nevs += k * H[k];
			});

			fK.use( (k) => {   
				if ( compress ) {
					if ( fK[k] ) K.push( k );
				}
				else
					K.push(k); 
			});

			var
				M = 0,
				Mdebug = 0,
				logfK = $(K.length, function (n,logf) {  // observed log count frequencies
					if ( Mdebug ) { // enables debugging
						logf[n] = logNB(K[n], Kbar, Mdebug);
						//logf[n] += (n%2) ? 0.5 : -0.5;  // add some "noise" for debugging
					}
					else
						logf[n] = fK[ K[n] ] ? log( fK[ K[n] ] ) : -7;
				});

			Log({
				Kbar: Kbar, 
				T: T, 
				N: N, 
				Kmax: Kmax,
				Nevs: Nevs,
				ci: [compress, interpolate]
			});

			if (false)
				K.use( (n) => {
					var k = K[n];
					Log(n, k, logNB(k,Kbar,55), logNB(k,Kbar,65), log( fK[k] ), logfK[n] );
				});

			if ( Kmax >= 2 ) {
				var M = {}, fits = {};

				if (solve.lma) {  // levenberg-marquadt algorithm for [M, ...]
					fits = LMA( solve.lma, K, logfK, logNB);
					M.lma = fits.parameterValues[0];
				}

				if (solve.lfa)   // linear factor analysis for M using newton-raphson search over chi^2. UAYOR !  (compression off, interpolation on)
					M.lfa = LFA( solve.lfa, fK, logNB);

				if (solve.bfs)  // brute force search for M
					M.bfs = BFS( solve.bfs, fK, logNB);

				var 
					M0 = M[solve.use || "lma"],
					snr = sqrt( Kbar / ( 1 + Kbar/M0 ) ),
					bias = sqrt( (Nevs-1)/2 ) * exp(GAMMA.log((Nevs-2)/2) - GAMMA.log((Nevs-1)/2)),		// bias of snr estimate
					mu = (Nevs>=4) ? (Nevs-1) / (Nevs-3) - bias**2 : 2.5;		// rel error in snr estimate

				cb({
					events: Nevs,
					est: M,
					fits: fits,
					coherence_intervals: M0,
					mean_count: Kbar,
					mean_intensity: Kbar / T,
					degeneracyParam: Kbar / M0,
					snr: snr,
					complete: 1 - mu/2.5,
					coherence_time: T / M0,
					fit_stats: M
				});
			}

			else
				cb( null );
		}
										
		const { sqrt, floor, random, cos, sin, abs, PI, log, exp} = Math;
		
		var
			flow = ctx.Flow;
		
		//Log("cints flow", flow);
		
		coherenceIntervals({  // define solver parms
			H: flow.F,		// count frequencies
			T: flow.T,  		// observation time
			N: flow.N,		// ensemble size
			use: ctx.Use,  // solution to retain
			lfa: ctx.lfa, // [50],  // initial guess at M = # coherence intervals
			bfs: ctx.bfs, // [1,200,5],  // M range and step to search
			lma: ctx.lma	// initial guess at M = # coherence intervals
		}, function (stats) {
			ctx.Save = stats;
			Log("cints save stats", stats);
			res(ctx);
		});

	}