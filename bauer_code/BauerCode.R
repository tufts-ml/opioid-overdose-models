data <- read.csv(paste0(Root, "test_opioid_mass.csv"), header=T)
data$timestep2 = data$timestep
m.1 <- inla(deaths_pred ~ f(geoid, model='iid') + f(timestep, model='rw1', constr = FALSE) + f(timestep2, model='ar1'), family='poisson', data=data, control.predictor = list(compute = TRUE, link = 1))
m.1$summary.fitted.values
linkna = rep(NA, n)
n = 9720
linkna = rep(NA, n)
link[which(is.na(deaths_pred))] = 1
linkna[which(is.na(deaths_pred))] = 1
linkna[which(is.na(data$deaths_pred))] = 1
m.1 <- inla(deaths_pred ~ f(geoid, model='iid') + f(timestep, model='rw1', constr = FALSE) + f(timestep2, model='ar1'), family='poisson', data=data, control.predictor = list(compute = TRUE, link = linkna))
m.1$summary.fitted.values
write.csv(m.1$summary.fitted.values, "./test_out.csv")
m.1$summary.fitted.values.is.na()
is.na(m.1$summary.fitted.values)
max(is.na(m.1$summary.fitted.values))
m.1$summary
summary(m.1)
m.1 <- inla(deaths_pred ~ f(geoid, model='iid') + f(timestep, model='rw1', constr = FALSE) + f(timestep2, model='ar1'), family='poisson', data=data, control.predictor = list(compute = TRUE, link = linkna))
m.1
m.1$summary.fitted.values
