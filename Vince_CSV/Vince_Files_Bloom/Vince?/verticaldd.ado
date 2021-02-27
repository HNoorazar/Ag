program define verticaldd
* version 2.0
*single sine heat unit accumulations with a vertical cutoff
*requires columns name max and min
*the program is called as verticaldd max min lower upper outname so for example
verticaldd max min 50 88 dd
quietly {
	args maxtemp mintemp lower upper outname

    tempvar summ diff diffsq b bsq a asq th1 th2 dd 
	
	local  twopi=2* _pi
	local  pihlf=0.5* _pi
	
*defines columns for later calculations
	generate double `summ'= `mintemp' + `maxtemp'
	generate double `diff'= `maxtemp' - `mintemp'
	generate double `diffsq'=`diff'^2
	


* this is 2*upper threshold - summ [fk3]

	generate double `b' = 2 * `upper'-`summ'
	generate double `bsq'=`b'^2

* this is 2*lower threshold - summ [fk1]
	generate double `a' = 2*`lower'-`summ'
	generate double `asq'=`a'^2
	generate double `th1'=atan(`a'/sqrt(`diffsq'-`asq'))
	generate double `th2'=atan(`b'/sqrt(`diffsq'-`bsq'))
	
*fourth decision min>=lower threshold and max>upper threshold     
	generate double `dd'=((-`diff'*cos(`th2')-`a'*(`th2' + `pihlf'))/`twopi') if `mintemp'>=`lower' & `maxtemp'>`upper'
	
*fifth major decision min>= lower threshold & `max' < upper threshold
	replace `dd'=`summ'/2-`lower' if `mintemp'>=`lower' & `maxtemp'<= `upper'

*sixth major decision if min<lower theshold & `maxtemp'< upper threshold
	replace `dd'=(`diff'*cos(`th1')-(`a'*(`pihlf'-`th1')))/`twopi' if `mintemp'<`lower' & `maxtemp'<= `upper'

*seventh major decision min<lower threshold & max>upper threshold
	replace `dd'=(-`diff'*(cos(`th2')-cos(`th1'))-(`a'*(`th2'-`th1')))/`twopi' if `mintemp'<`lower' & `maxtemp'>`upper'

*decisions 1-3
	replace `dd'=0 if `mintemp'>`maxtemp' | `maxtemp'<=`lower' | `mintemp'>=`upper'
*move dd to the right place & format it
generate `outname'=`dd'
	format `outname' %9.1f
	
}
end

	
