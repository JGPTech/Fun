Projectile Motion with Position-Dependent Acceleration

When acceleration depends on **position** (e.g. gravity that varies with height), you
can’t just integrate with respect to time directly. Instead, reduce the system using
the chain rule:

\[
a_y = \frac{dv_y}{dt} = \frac{dv_y}{dy}\frac{dy}{dt} = v_y \frac{dv_y}{dy}.
\]

This trick turns a time-dependent ODE into a separable one in \(y\).

---

## 1. General Setup

For projectile motion:

\[
\begin{aligned}
x''(t) &= 0, \\
y''(t) &= g(y).
\end{aligned}
\]

Vertical motion reduction:

\[
v_y \frac{dv_y}{dy} = g(y).
\]

Integrate once:

\[
\frac{1}{2}v_y^2 = \int g(y)\,dy + C.
\]

With initial condition \(v_y(y_0)=v_{y0}\):

\[
\frac{1}{2}v_y^2 = \int_{y_0}^{y} g(\eta)\,d\eta + \frac{1}{2}v_{y0}^2.
\]

Then recover time:

\[
\frac{dy}{dt}=v_y(y) \quad\Rightarrow\quad
t(y)=\int_{y_0}^{y}\frac{d\eta}{v_y(\eta)}.
\]

Horizontal motion is simple:

\[
x(t)=x_0+v_{x0}\,t.
\]

---

## 2. Examples

### A. Constant Gravity (\(g(y)=-g_0\))

\[
\frac{1}{2}v_y^2 = -g_0(y-y_0)+\frac{1}{2}v_{y0}^2.
\]

So

\[
v_y(y)=\pm\sqrt{v_{y0}^2 - 2g_0(y-y_0)}.
\]

Integrating for time gives the familiar kinematic equations:

\[
y(t)=y_0+v_{y0}t-\tfrac12 g_0 t^2, \quad x(t)=x_0+v_{x0}t.
\]

---

### B. Inverse-Square Gravity

\[
g(y) = -\frac{GM}{(R+y)^2}.
\]

Then

\[
\frac{1}{2}v_y^2 = GM\!\left(\frac{1}{R+y}-\frac{1}{R+y_0}\right) + \frac{1}{2}v_{y0}^2.
\]

So

\[
v_y(y) = \pm\sqrt{v_{y0}^2 + 2GM\!\left(\frac{1}{R+y}-\frac{1}{R+y_0}\right)}.
\]

Time as a function of height:

\[
t(y)=\int_{y_0}^{y}\frac{d\eta}{\sqrt{v_{y0}^2 + 2GM\!\left(\frac{1}{R+\eta}-\frac{1}{R+y_0}\right)}}.
\]

Horizontal:

\[
x(t)=x_0+v_{x0}t.
\]

---

## 3. How to Set Up the Differential Equations

1. Pick your force law:  
   \(\;F_y(y) = m\,g(y)\).

2. Write the ODE:  
   \(\;y'' = g(y).\)

3. Apply the reduction:  
   \(\;v_y\,dv_y/dy = g(y).\)

4. Integrate to get \(v_y(y)\).

5. Integrate again to get \(t(y)\), then combine with \(x(t)\).

---

### Key Takeaway

Whenever acceleration depends on **position**, reduce via

\[
a = v\,\frac{dv}{dy}.
\]

That’s the doorway to solvable integrals.
