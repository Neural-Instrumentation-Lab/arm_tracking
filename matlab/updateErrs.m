function c = updateErrs(err,c)

c.wts = c.wts - (c.beta * err * c.p);
