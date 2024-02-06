# Simulation of the fractional coalescent with two populations

The simulation code `simtree.py` will be available in our public GitHub. Here, a few examples of the output are shown. The key code is given in the snippet in Figure [1](#fig1). Figure [2](#fig2) gives examples for two populations with different $\alpha$, all histograms were drawn from 5000 independent replicates using the same effective population sizes ($\Theta_1=0.01$,$\Theta_2=0.01$) and immigration rates ($M_{2\rightarrow1}=100$, $M_{1\rightarrow2}=100$), but different $\alpha$. Each histogram is compared with the standard Kingman coalescent.

<figure id="fig1">
<div class="sourceCode" id="cb1"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="co"># generates Mittag-Leffler time interval based on mylambda and alpha</span></span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="co"># for each evolutionary force: Theta_1, Theta_2, M_21, M_12</span></span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="co"># for the future, I assume this also will work for growth and population divergence </span></span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="co"># with the correct lambda </span></span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> randommlftime(mylambda, alpha):</span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a>    pia <span class="op">=</span> <span class="fl">3.1415926</span> <span class="op">*</span> alpha</span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a>    r1 <span class="op">=</span> np.random.uniform(<span class="dv">0</span>,<span class="dv">1</span>)</span>
<span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a>    r2 <span class="op">=</span> np.random.uniform(<span class="dv">0</span>,<span class="dv">1</span>)</span>
<span id="cb1-9"><a href="#cb1-9" aria-hidden="true" tabindex="-1"></a>    denoma <span class="op">=</span> <span class="fl">1.0</span> <span class="op">/</span> alpha</span>
<span id="cb1-10"><a href="#cb1-10" aria-hidden="true" tabindex="-1"></a>    denomlambda <span class="op">=</span> <span class="fl">1.0</span> <span class="op">/</span> mylambda</span>
<span id="cb1-11"><a href="#cb1-11" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> <span class="op">-</span>denomlambda<span class="op">**</span>denoma <span class="op">*</span> </span>
<span id="cb1-12"><a href="#cb1-12" aria-hidden="true" tabindex="-1"></a>         (np.sin(pia)<span class="op">/</span>(np.tan(pia<span class="op">*</span>(<span class="fl">1.0</span><span class="op">-</span>r1)))<span class="op">-</span>np.cos(pia))<span class="op">**</span>denoma <span class="op">*</span> </span>
<span id="cb1-13"><a href="#cb1-13" aria-hidden="true" tabindex="-1"></a>         np.log(r2)</span>
<span id="cb1-14"><a href="#cb1-14" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-15"><a href="#cb1-15" aria-hidden="true" tabindex="-1"></a><span class="co"># creates the time for a migration or coalescent event,</span></span>
<span id="cb1-16"><a href="#cb1-16" aria-hidden="true" tabindex="-1"></a><span class="co"># evaluating the time intervals for each force and picks the smallest</span></span>
<span id="cb1-17"><a href="#cb1-17" aria-hidden="true" tabindex="-1"></a><span class="co"># looping through Y , the alphas vector here needs and entry for every force</span></span>
<span id="cb1-18"><a href="#cb1-18" aria-hidden="true" tabindex="-1"></a><span class="co"># see the function fill_Yalphas()                                                                                                 </span></span>
<span id="cb1-19"><a href="#cb1-19" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> randomtime(Y,alphas,t0):</span>
<span id="cb1-20"><a href="#cb1-20" aria-hidden="true" tabindex="-1"></a>    smallu <span class="op">=</span> <span class="fl">1e100</span></span>
<span id="cb1-21"><a href="#cb1-21" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> yi,ai <span class="kw">in</span> <span class="bu">zip</span>(Y,alphas):</span>
<span id="cb1-22"><a href="#cb1-22" aria-hidden="true" tabindex="-1"></a>        u <span class="op">=</span> randommlftime(yi,ai)</span>
<span id="cb1-23"><a href="#cb1-23" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> u <span class="op">&lt;</span> smallu:</span>
<span id="cb1-24"><a href="#cb1-24" aria-hidden="true" tabindex="-1"></a>            smallu <span class="op">=</span> u</span>
<span id="cb1-25"><a href="#cb1-25" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> t0 <span class="op">+</span> smallu</span></code></pre></div>
<figcaption>Key python function to draw new event times using the
fractional coalescent with different <span
class="math inline"><em>α</em></span> per population.</figcaption>
</figure>

<figure id="fig2">
<p><img src="simtree-0.999-0.999.pdf" style="width:32.0%" alt="image" />
<img src="simtree-0.999-0.9.pdf" style="width:32.0%" alt="image" /> <img
src="simtree-0.999-0.7.pdf" style="width:32.0%" alt="image" /><br />
<img src="simtree-0.9-0.9.pdf" style="width:32.0%" alt="image" /> <img
src="simtree-0.9-0.999.pdf" style="width:32.0%" alt="image" /> <img
src="simtree-0.7-0.999.pdf" style="width:32.0%" alt="image" /></p>
<figcaption>Comparison of six different scenarios for two populations
with different <span class="math inline"><em>α</em></span>: top left:
both populations are essentially following the Kingman coalescent; top
middle and right: one population deviates from the Kingman coalescent;
bottom left: both populations deviate similarly from Kingman coalescent,
bottom middle and right: Same scenario as ‘top middle and right’ except
that the <span class="math inline"><em>α</em></span> are
reversed.</figcaption>
</figure>

::: samepage
The script has several options:

    usage: simtree.py [-h] [-l LOCI] 
                                        [-s SITES] 
                                        [-i INDIVIDUALS] 
                                        [-t THETA]
                                         [-m MIG] 
                                         [-a ALPHA] 
                                         [-f FILE] 
                                         [-p]

    Simulate a tree

    optional arguments:
      -h, --help            show this help message and exit
      -l LOCI, --loci LOCI  number of loci
      -s SITES, --sites SITES
                            number of sites
      -i INDIVIDUALS, --individuals INDIVIDUALS
                            Number of samples for each population
      -t THETA, --theta THETA
                            thetas for each population
      -m MIG, --mig MIG     migration rate for each population
      -a ALPHA, --alpha ALPHA
                            alpha for each population
      -f FILE, --file FILE  treefile to be used with migdata, default is NONE 
                                                which is a placeholder for sys.stdout
      -p, --plot            Plots density histogram of TMRCA
:::
