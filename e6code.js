function xss(ctx,res) {
		res([
			{lat:33.902,lon:70.09,alt:22,t:10},
			{lat:33.902,lon:70.09,alt:12,t:20}
		]);
	}e.Stats_Gain = assumed detector gain = area under trigger function
		Stats.coherence_time = coherence time underlying the process
		Dim = samples in profile = max coherence intervals
		Flow.T = observation time
		Events = query to get events
	*/
		const { sqrt, floor, random, cos, sin, abs, PI, log, exp} = Math;
		
		function triggerProfile( solve, cb) {
		/**
		Use the Paley-Wiener Theorem to return the trigger function stats:

			x = normalized time interval of recovered trigger
			h = recovered trigger function at normalized times x
			modH = Fourier modulous of recovered trigger at frequencies f
			argH = Fourier argument of recovered trigger at frequencies f
			f = spectral frequencies

		via the callback cb(stats) given a solve request:

			evs = events list
			refLambda = ref mean arrival rate (for debugging)
			alpha = assumed detector gain
			N = profile sample times = max coherence intervals
			model = correlation model name
			Tc = coherence time of arrival process
			T = observation time
		*/
			
			var 
				ctx = {
					evs: ME.matrix( solve.evs ),
					N: solve.N,
					refLambda: solve.refLambda,
					alpha: solve.alpha,
					T: solve.T,
					Tc: solve.Tc
				},
				script = `
N0 = fix( (N+1)/2 );
fs = (N-1)/T;
df = fs/N;
nu = rng(-fs/2, fs/2, N); 
t = rng(-T/2, T/2, N); 
V = evpsd(evs, nu, T, "n", "t");  

Lrate = V.rate / alpha;
Accf = Lrate * ${solve.model}(t/Tc);
Lccf = Accf[N0]^2 + abs(Accf).^2;
Lpsd =  wkpsd( Lccf, T);
disp({ 
	evRates: {ref: refLambda, ev: V.rate, L0: Lpsd[N0]}, 
	idx0lag: N0, 
	obsTime: T, 
	sqPower: {N0: N0, ccf: Lccf[N0], psd: sum(Lpsd)*df }
});

Upsd = Lrate + Lpsd;
modH = sqrt(V.psd ./ Upsd );  

argH = pwt( modH, [] ); 
h = re(dft( modH .* exp(i*argH),T)); 
x = t/T; 
`;
			ME.exec(script,  ctx, function (ctx) {
				//Log("vmctx", ctx);
				cb({
					trigger: {
						x: ctx.x,
						h: ctx.h,
						modH: ctx.modH,
						argH: ctx.argH,
						f: nu
					}
				});
			});
		}
		
		var
			stats = ctx.Stats,
			file = ctx.File,
			flow = ctx.Flow;
		
		if (stats.coherence_time)
			FLOW.all(ctx, function (evs) {  // fetch all the events
				if (evs)
					triggerProfile({  // define solver parms
						evs: evs,		// events
						refLambda: stats.mean_intensity, // ref mean arrival rate (for debugging)
						alpha: file.Stats_Gain, // assumed detector gain
						N: ctx.Dim, 		// samples in profile = max coherence intervals
						model: ctx.Model,  	// name correlation model
						Tc: stats.coherence_time,  // coherence time of arrival process
						T: flow.T  		// observation time
					}, function (stats) {
						ctx.Save = stats;
						res(ctx);
					});
			});
		
		else
			res(null);
	}}],
						function (err, pcs) {
							if (!err) cb(pcs);
					});
				}

				findpcs( function (pcs) {
					if (pcs.length) 
						sendpcs( pcs );

					else
					SQL.query(
						"SELECT count(ID) as Count FROM app.pcs WHERE least(?,1)", {
							max_intervals: Mmax, 
							correlation_model: model
						}, 
						function (err, test) {  // see if pc model exists

						//Log("test", test);
						if ( !test[0].Count )  // pc model does not exist so make it
							genpcs( Mmax, Mwin*2, model, function () {
								findpcs( sendpcs );
							});

						else  // search was too restrictive so no need to remake model
							sendpcs(pcs);
					});							
				});
			}
	
			getpcs( solve.model||"sinc", solve.min||0, solve.M, solve.Mstep/2, solve.Mmax, function (pcs) {
				
				const { sqrt, random, log, exp, cos, sin, PI } = Math;
				
				function expdev(mean) {
					return -mean * log(random());
				}
				
				if (pcs) {
					var 
						pcRef = pcs.ref,  // [unitless]
						pcVals = pcs.values,  // [unitless]
						N = pcVals.length,
						T = solve.T,
						dt = T / (N-1),
						egVals = $(N, (n,e) => e[n] = solve.lambdaBar * dt * pcVals[n] * pcRef ),  // [unitless]
						egVecs = pcs.vectors,   // [sqrt Hz]
						ctx = {
							T: T,
							N: N,
							dt: dt,
							
							E: ME.matrix( egVals ),
							
							B: $(N, (n,B) => {
								var
									b = sqrt( expdev( egVals[n] ) ),  // [unitless]
									arg = random() * PI;

								Log(n,arg,b, egVals[n], T, N, solve.lambdaBar );
								B[n] = ME.complex( b * cos(arg), b * sin(arg) );  // [unitless]
							}),

							V: egVecs   // [sqrt Hz]
						},
						script = `
A=B*V; 
lambda = abs(A).^2 / dt; 
Wbar = {evd: sum(E), prof: sum(lambda)*dt};
evRate = {evd: Wbar.evd/T, prof: Wbar.prof/T};
x = rng(-1/2, 1/2, N); 
`;

//Log(ctx);

					if (N) 
						ME.exec( script , ctx, (ctx) => {
							//Log("ctx", ctx);
							cb({
								intensity: {x: ctx.x, i: ctx.lambda},
								//mean_count: ctx.Wbar,
								//mean_intensity: ctx.evRate,
								eigen_ref: pcRef
							});
							Log({
								mean_count: ctx.Wbar,
								mean_intensity: ctx.evRate,
								eigen_ref: pcRef
							});
						});	

					else
						cb({
							error: `coherence intervals ${stats.coherence_intervals} > max pc dim`
						});
				}

				else
					cb({
						error: "no pcs matched"
					});
			});
		}

		var
			stats = ctx.Stats,
			flow = ctx.Flow;
		
		Log("rats ctx", stats);
		if (stats)
			arrivalRates({  // parms for principle components (intensity profile) solver
				trace: false,   // eigen debug
				T: flow.T,  // observation interval  [1/Hz]
				M: stats.coherence_intervals, // coherence intervals
				lambdaBar: stats.mean_intensity, // event arrival rate [Hz]
				Mstep: 1,  // coherence step size when pc created
				Mmax: ctx.Dim || 150,  // max coherence intervals when pc created
				model: ctx.Model,  // assumed correlation model for underlying CCGP
				min: ctx.MinEigen	// min eigen value to use
			}, function (stats) {
				ctx.Save = stats;
				Log("save", stats);
				res(ctx);
			});
		
		else
			res(null);
		
	}
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