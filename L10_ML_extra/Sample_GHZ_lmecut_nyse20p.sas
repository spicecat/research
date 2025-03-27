/************************ Settings for local processing *************************/
libname home "D:\Research\Data\Turnover_Penalty";
libname ghz "D:\Research\Data\Anomaly_Vars\GHZ";
libname crsp "D:\Research\Data\Raw_Data\CRSP";
libname factor "D:\Research\Data\Var";
%include "D:\Research\Function\crspmerge.sas";
%include "D:\Research\Function\nwords.sas";

/* Characteristics variables */
/* GHZ 102 variables */
%let CharsVars = absacc acc aeavol age agr baspread beta betasq bm bm_ia
                 cash cashdebt cashpr cfp cfp_ia chatoia chcsho chempia chfeps chinv
                 chmom chnanalyst chpmia chtx cinvest convind currat depr disp divi
                 divo dolvol dy ear egr ep fgr5yr gma grcapx grltnoa
                 herf hire idiovol ill indmom invest ipo lev lgr maxret
                 mom12m mom1m mom36m mom6m ms mve mve_ia nanalyst nincr operprof
                 orgcap pchcapx_ia pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv
                 pctacc pricedelay ps quick rd rd_mve rd_sale realestate retvol roaq
		 		 roavol roeq roic rsup salecash saleinv salerec secured securedind sfe
                 sgr sin sp std_dolvol std_turn stdacc stdcf sue tang tb
				 turn zerotrade;
%let outfile = D:\Research\Data\Turnover_Penalty;

/********************************************************************************/
/******************************* 0. Generate code *******************************/
%Macro gencode(CharsVars);
%global code_sql;
%let code_sql = %str();
%let nvars = %nwords(&CharsVars.);
%do i = 1 %to &nvars.;
    %let cvar_i = %scan(&CharsVars., &i., %str( ));
	%let code_sql = &code_sql. b.&cvar_i.;
	%if &i. ne &nvars. %then %do;
        %let code_sql = &code_sql.,;
	%end;
%end;
%Mend;

%gencode(&CharsVars.);
%put &code_sql.;

/********************************************************************************/
/******************************* 1. CRSP monthly ********************************/
%let startdate = 01jan1960;
%let enddate = 31dec2019;
%let sfilter = shrcd in (10,11) and exchcd in (1,2,3);
%let msevars = ncusip exchcd shrcd siccd dlstcd dlret;
%let msfvars = permco prc ret retx shrout vol cusip cfacpr cfacshr;
%CrspMerge(s=m, start=&startdate, end=&enddate, sfvars=&msfvars, sevars=&msevars, filters=&sfilter);

data crsp_m;
  set crsp_m;
  /* Time variable */
  date = intnx('month', date, 0, 'E');
  ym = (year(date)-1960)*12 + month(date)-1;

  /* Price & Market equity */
  prc = abs(prc);
  me = prc*shrout/1000;

  /* Adjust missing delisting returns */
  /* Shumway (1997) suggests to use -0.3 to fill the missing performance delisting returns in CRSP */
  /* Shumway and Warther (1999) suggest to use -0.55 to fill the missing performance delisting returns in CRSP's Nasdaq data */
  /* The performance category: CRSP delisting codes 500 and 520-584 */
  if (dlstcd=500 | dlstcd<=584 and dlstcd>=520) & exchcd=3 & (dlret=.S | dlret=.T | dlret=.P) then ddlret = -0.55;
  else if (dlstcd=500 | dlstcd<=584 and dlstcd>=520) & (exchcd=1 | exchcd=2) & (dlret=.S | dlret=.T | dlret=.P) then ddlret = -0.3;
  else if dlret=.S | dlret=.T | dlret=.A | dlret=.P then ddlret = .;
  else ddlret = dlret;
  
  /* WRDS: (1+DLRET)*(1+RET)-1 */
  if dlstcd=. or dlstcd=100 then retadj = ret;
  else if nmiss(ret,ddlret) ne 2 then retadj = sum(1,ret)*sum(1,ddlret) - 1;
  else retadj = .;
run;

/********************************************************************************/
/********************************* 2. GHZ data **********************************/
data ghz_data0;
  set ghz.rpsdata_rfs (keep = permno date &CharsVars.);
  date = intnx('month', date, 0, 'E');
  ym = (year(date)-1960)*12 + month(date)-1;
run;

/********************************** Winsorize ***********************************/
/********************** Exclude dummy and score variables ***********************/
/* Dummy: convind, divi, divo, ipo, rd, securedind, sin                         */
/* Score: ms, nincr, ps                                                         */
/********************************************************************************/
%let WinsorVars = absacc acc aeavol age agr baspread beta betasq bm bm_ia
                  cash cashdebt cashpr cfp cfp_ia chatoia chcsho chempia chfeps chinv
                  chmom chnanalyst chpmia chtx cinvest currat depr disp
                  dolvol dy ear egr ep fgr5yr gma grcapx grltnoa
                  herf hire idiovol ill indmom invest lev lgr maxret
                  mom12m mom1m mom36m mom6m mve mve_ia nanalyst operprof
                  orgcap pchcapx_ia pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv
                  pctacc pricedelay quick rd_mve rd_sale realestate retvol roaq
		 		  roavol roeq roic rsup salecash saleinv salerec secured sfe
                  sgr sp std_dolvol std_turn stdacc stdcf sue tang tb
				  turn zerotrade;

%Macro gencode_winsor(WinsorVars);
%global WinsorVars_P1 WinsorVars_P99 code_winsor;
%let WinsorVars_P1 = %str();
%let WinsorVars_P99 = %str();
%let code_winsor = %str();
%let nvars = %nwords(&WinsorVars.);
%do i = 1 %to &nvars.;
    %let wvar_i = %scan(&WinsorVars., &i., %str( ));
	%let WinsorVars_P1 = &WinsorVars_P1. &wvar_i._P1;
	%let WinsorVars_P99 = &WinsorVars_P99. &wvar_i._P99;
	%let code_winsor = &code_winsor. b.&wvar_i._P1, b.&wvar_i._P99;
	%if &i. ne &nvars. %then %do;
        %let code_winsor = &code_winsor.,;
	%end;
%end;
%Mend;

%gencode_winsor(&WinsorVars.);
%put &WinsorVars_P1.;
%put &WinsorVars_P99.;
%put &code_winsor.;

/* No dupkey */
proc sort nodupkey data = ghz_data0; by ym permno; run;
proc means noprint
  data = ghz_data0;
  by ym;
  var &WinsorVars.;
  output out=ghz_quantile min= p1= p99= max= /autoname;
run;

proc sql;
  create table ghz_data as
  select a.*, &code_winsor.
  from ghz_data0 as a
  left join ghz_quantile as b
  on a.ym = b.ym;
quit;

data ghz_data;
  set ghz_data;
  array base {*} &WinsorVars.;
  array varl {*} &WinsorVars_P1.;
  array varh {*} &WinsorVars_P99.;
  do i = 1 to dim(base);
	 if (not missing(base(i))) and (not missing(varl(i))) and (base(i)<varl(i)) then base(i) = varl(i);
	 if (not missing(base(i))) and (not missing(varh(i))) and (base(i)>varh(i)) then base(i) = varh(i);
  end;
  drop i &WinsorVars_P1. &WinsorVars_P99.;
run;

proc means data = ghz_data0; var &WinsorVars.; run;
proc means data = ghz_data; var &WinsorVars.; run;
proc sql; drop table ghz_data0; quit;

/********************************************************************************/
/****************************** 3. Merge databases ******************************/
/* Factor */
data factor_3m; set factor.factor_3m; run;

/* Me breakpoints */
data me_breakpoints; set factor.me_breakpoints; run;

/* Lagged me & risk-free rate & me breakpoints */
proc sql;
  create table sample_ghz0 as
  select a.permno, a.date, a.ym, a.me, b.me as lme, a.retadj, a.retadj - c.rf/100 as exret, c.rf/100 as rf,
         d.p20 as lme_cutpoint, d.date as date_cutpoint
  from crsp_m as a
  left join crsp_m (keep = permno ym me) as b
  on a.permno = b.permno and a.ym = b.ym + 1
  left join factor_3m as c
  on a.ym = c.ym
  left join me_breakpoints as d
  on a.ym = d.ym + 1;
quit;

/* No dupkey */
proc sort nodupkey data = sample_ghz0; by ym permno; run;

/* Lagged me rank */
proc rank
  data = sample_ghz0 out = sample_ghz0;
  by ym;
  var lme;
  ranks lme_norm_rk;
run;

proc sql;
  create table sample_ghz1 (drop = lme_norm_rk) as
  select *, (lme_norm_rk - min(lme_norm_rk)) / (max(lme_norm_rk) - min(lme_norm_rk)) as lme_norm
  from sample_ghz0
  group by ym
  order by ym, permno;
quit;

/* Save */
data tcost_info;
  set sample_ghz1 (keep = permno ym lme lme_norm retadj exret);
run;

proc export
  data = tcost_info
  outfile = "&outfile.\tcost_info_nyse20p.dta"
  dbms = stata replace;
run;

/* Delete the obs with no GHZ chars */
/* Thus, use inner join rather than left join */
/* Read 3,244,865 obs, keep 2,526,036 obs */
proc sql;
  create table sample_ghz as
  select a.*, &code_sql.
  from sample_ghz1 as a
  inner join ghz_data as b
  on a.permno = b.permno and a.ym = b.ym;
quit;
proc sql; drop table sample_ghz1; quit;

/* No dupkey */
proc sort nodupkey data = sample_ghz; by ym permno; run;

/*********************************** Filters ************************************/
/* Keep 1,074,416 obs */
data sample_ghz;
  set sample_ghz;
  if missing(retadj) then delete;
  if missing(lme) then delete;
  if lme < lme_cutpoint then delete;
  drop lme_cutpoint date_cutpoint;
run;

/* Check */
/* No missing exret */
data ck;
  set sample_ghz;
  where missing(exret);
run;

/* Histogram */
proc univariate
  data = sample_ghz;
  var lme_norm;
  histogram;
run;

/*********************************** 4. Save ************************************/
proc sort data = sample_ghz; by ym permno; run;
proc export
  data = sample_ghz
  outfile = "&outfile\sample_ghz_nyse20p.dta"
  dbms = stata replace;
run;
