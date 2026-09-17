import type * as React from 'react'

import { cn } from '@/lib/utils'

function Card({ className, ...props }: React.ComponentProps<'section'>) {
  return <section data-slot="card" className={cn('rounded-xl border bg-card text-card-foreground', className)} {...props} />
}

function CardHeader({ className, ...props }: React.ComponentProps<'header'>) {
  return <header className={cn('flex items-center justify-between gap-3 px-5 pt-4 pb-3', className)} {...props} />
}

function CardTitle({ className, ...props }: React.ComponentProps<'h2'>) {
  return (
    <h2
      className={cn('text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase', className)}
      {...props}
    />
  )
}

function CardContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('px-5 pb-5', className)} {...props} />
}

export { Card, CardContent, CardHeader, CardTitle }
