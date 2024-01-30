# Load the necessary library
library(tree)
if (!require("rare")) {
    install.packages("rare")
    library("rare")
}

tree_matrix <- function(hc) {
    p <- nrow(hc$merge) + 1
    # print(p)
    n_interior <- nrow(hc$merge)
    A_i <- c(as.list(seq(p)), sapply(seq(n_interior), function(x) {
        find.leaves(
            x,
            hc$merge
        )
    }))
    # print(hc$merge)
    A_j <- sapply(seq(length(A_i)), function(x) rep(x, len = length(A_i[[x]])))
    # A <- sparseMatrix(i = unlist(A_i), j = unlist(A_j), x = rep(1,
    #     len = length(unlist(A_i))
    # ))
    # A
}

generate_tree <- function(
    formula, data, weights, subset, na.action = na.pass,
    control = tree.control(nobs, ...), method = "recursive.partition",
    split = c("deviance", "gini"), model = FALSE, x = FALSE,
    y = TRUE, wts = TRUE, ...) {
    if (is.data.frame(model)) {
        m <- model
        model <- FALSE
    } else {
        m <- match.call(expand.dots = FALSE)
        m$method <- m$model <- m$control <- m$... <- m$x <- m$y <- m$wts <- m$split <- NULL
        m[[1L]] <- as.name("model.frame.default")
        m <- eval.parent(m)
        if (method == "model.frame") {
            return(m)
        }
    }
    split <- match.arg(split) # deviance
    Terms <- attr(m, "terms")
    if (any(attr(Terms, "order") > 1)) {
        stop("trees cannot handle interaction terms")
    }
    Y <- stats::model.extract(m, "response")
    if (is.matrix(Y) && ncol(Y) > 1L) {
        stop("trees cannot handle multiple responses")
    }
    ylevels <- levels(Y)
    w <- stats::model.extract(m, "weights")
    if (!length(w)) {
        w <- rep(1, nrow(m))
    }
    if (any(yna <- is.na(Y))) {
        Y[yna] <- 1
        w[yna] <- 0
    }
    offset <- attr(Terms, "offset")
    if (!is.null(offset)) {
        if (length(ylevels)) {
            stop("offset not implemented for classification trees")
        }
        offset <- m[[offset]]
        Y <- Y - offset
    }
    X <- rare::tree.matrix(m)
    xlevels <- attr(X, "column.levels")
    if (is.null(xlevels)) {
        xlevels <- rep(list(NULL), ncol(X))
        names(xlevels) <- dimnames(X)[[2L]]
    }
    nobs <- length(Y)
    if (nobs == 0L) {
        stop("no observations from which to fit a model")
    }
    if (!is.null(control$nobs) && control$nobs < nobs) {
        stop("control$nobs < number of observations in data")
    }
    fit <- .C("BDRgrow1", as.double(X), as.double(unclass(Y)),
        as.double(w), as.integer(c(sapply(xlevels, length), length(ylevels))),
        as.integer(rep(1, nobs)), as.integer(nobs), as.integer(ncol(X)),
        node = integer(control$nmax), var = integer(control$nmax),
        cutleft = character(control$nmax), cutright = character(control$nmax),
        n = double(control$nmax), dev = double(control$nmax),
        yval = double(control$nmax), yprob = double(max(control$nmax *
            length(ylevels), 1)), as.integer(control$minsize),
        as.integer(control$mincut), as.double(max(0, control$mindev)),
        nnode = as.integer(0L), where = integer(nobs), as.integer(control$nmax),
        as.integer(split == "gini"), as.integer(sapply(m, is.ordered)),
        NAOK = TRUE
    )
    n <- fit$nnode
    frame <- data.frame(fit[c("var", "n", "dev", "yval")])[1L:n, ]
    frame$var <- factor(frame$var, 0:length(xlevels), c(
        "<leaf>",
        names(xlevels)
    ))
    frame$splits <- array(
        unlist(fit[c("cutleft", "cutright")]),
        c(control$nmax, 2), list(character(0L), c(
            "cutleft",
            "cutright"
        ))
    )[1L:n, , drop = FALSE]
    if (length(ylevels)) {
        frame$yval <- factor(
            frame$yval, 1L:length(ylevels),
            ylevels
        )
        class(frame$yval) <- class(Y)
        frame$yprob <- t(array(fit$yprob, c(
            length(ylevels),
            control$nmax
        ), list(ylevels, character(0L)))[, 1L:n,
            drop = FALSE
        ])
    }
    row.names(frame) <- fit$node[1L:n]
    fit <- list(
        frame = frame, where = fit$where, terms = Terms,
        call = match.call()
    )
    attr(fit$where, "names") <- row.names(m)
    if (n > 1L) {
        class(fit) <- "tree"
    } else {
        class(fit) <- c("singlenode", "tree")
    }
    attr(fit, "xlevels") <- xlevels
    if (length(ylevels)) {
        attr(fit, "ylevels") <- ylevels
    }
    if (is.logical(model) && model) {
        fit$model <- m
    }
    if (x) {
        fit$x <- X
    }
    if (y) {
        fit$y <- Y
    }
    if (wts) {
        fit$weights <- w
    }
    fit
}

Boston <- read.csv("boston.csv")
included_rows <- c(
    505, 324, 167, 129, 418, 471,
    299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
Boston <- Boston[included_rows, c("CRIM", "ZN", "INDUS", "Price")]
set.seed(1)
tree_model <- generate_tree(Price ~ ., data = Boston)
tree.boston <- tree(Price ~ ., data = Boston)

# Print the tree model
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty = 0)
